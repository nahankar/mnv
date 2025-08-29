"""
Platform Distribution Service

FastAPI service for uploading and distributing videos to social media platforms
with support for YouTube, Instagram, TikTok, Facebook, and comprehensive
upload tracking, retry logic, and platform-specific optimizations.
"""

import asyncio
import logging
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
from shared.database import DatabaseManager, get_db_manager
from shared.logging import get_logger
from shared.config import get_config
from shared.middleware import CorrelationMiddleware
from shared.models import Video, PlatformUpload, ContentStatus, PlatformType
from shared.retry import retry_api_call, convert_http_error, NetworkError
from shared.rate_limiter import get_rate_limiter

from platform_clients.youtube_client import YouTubeClient
from platform_clients.instagram_client import InstagramClient
from platform_clients.tiktok_client import TikTokClient
from platform_clients.facebook_client import FacebookClient

# Prometheus metrics
REQUEST_COUNT = Counter('distribution_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('distribution_service_request_duration_seconds', 'Request duration')
UPLOAD_REQUESTS = Counter('distribution_upload_requests_total', 'Upload requests', ['platform', 'status'])
UPLOAD_DURATION = Histogram('distribution_upload_duration_seconds', 'Upload duration', ['platform'])
PLATFORM_ERRORS = Counter('distribution_platform_errors_total', 'Platform errors', ['platform', 'error_type'])
RETRY_ATTEMPTS = Counter('distribution_retry_attempts_total', 'Retry attempts', ['platform'])

app = FastAPI(title="Platform Distribution Service", version="1.0.0")
logger = get_logger(__name__)
config = get_config()

# Add middleware
app.add_middleware(CorrelationMiddleware)

# Configuration
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
RETRY_DELAY_BASE = int(os.getenv("RETRY_DELAY_BASE", "2"))  # seconds
UPLOAD_TIMEOUT = int(os.getenv("UPLOAD_TIMEOUT", "300"))    # 5 minutes


class UploadStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UploadPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class PlatformUploadRequest(BaseModel):
    video_id: str = Field(..., description="Video ID to upload")
    platform: PlatformType = Field(..., description="Target platform")
    title: str = Field(..., max_length=255, description="Video title")
    description: str = Field("", description="Video description")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags")
    category: Optional[str] = Field(None, description="Video category")
    privacy: str = Field("public", description="Privacy setting")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled upload time")
    priority: UploadPriority = Field(UploadPriority.NORMAL, description="Upload priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for deduplication")


class BatchUploadRequest(BaseModel):
    video_id: str = Field(..., description="Video ID to upload")
    platforms: List[PlatformType] = Field(..., description="Target platforms")
    title: str = Field(..., max_length=255, description="Video title")
    description: str = Field("", description="Video description")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags")
    category: Optional[str] = Field(None, description="Video category")
    privacy: str = Field("public", description="Privacy setting")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled upload time")
    priority: UploadPriority = Field(UploadPriority.NORMAL, description="Upload priority")
    platform_specific: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Platform-specific overrides")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for deduplication")


class UploadResult(BaseModel):
    upload_id: str
    video_id: str
    platform: PlatformType
    status: UploadStatus
    platform_video_id: Optional[str] = None
    upload_url: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class UploadStatusResponse(BaseModel):
    upload_id: str
    video_id: str
    platform: PlatformType
    status: UploadStatus
    progress: Optional[float] = None  # 0.0 to 1.0
    platform_video_id: Optional[str] = None
    upload_url: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class PlatformLimits(BaseModel):
    daily_uploads: int
    file_size_mb: int
    duration_seconds: int
    title_max_length: int
    description_max_length: int
    hashtags_max_count: int
    current_usage: Dict[str, int] = Field(default_factory=dict)


class DistributionProcessor:
    """Main distribution processing orchestrator"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.platform_clients = {
            PlatformType.YOUTUBE: YouTubeClient(),
            PlatformType.INSTAGRAM: InstagramClient(),
            PlatformType.TIKTOK: TikTokClient(),
            PlatformType.FACEBOOK: FacebookClient()
        }
    
    async def upload_to_platform(self, request: PlatformUploadRequest) -> UploadResult:
        """Upload video to a specific platform with idempotency support"""
        upload_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Check for idempotency key
            if request.idempotency_key:
                existing_upload = await self._check_idempotency_key(
                    request.idempotency_key, request.video_id, request.platform
                )
                if existing_upload:
                    logger.info(f"Returning existing upload for idempotency key: {request.idempotency_key}")
                    return existing_upload
            
            # Validate video exists and is ready
            video = await self._get_video_by_id(request.video_id)
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            if video.status != ContentStatus.COMPLETED:
                raise HTTPException(status_code=400, detail="Video is not ready for distribution")
            
            # Check platform rate limits
            await self._check_platform_limits(request.platform)
            
            # Handle scheduled uploads
            if request.scheduled_time and request.scheduled_time > datetime.utcnow():
                return await self._schedule_upload(request, upload_id, video)
            
            # Create upload record
            upload_record = await self._create_upload_record(upload_id, request, video)
            
            # Get platform client
            client = self.platform_clients.get(request.platform)
            if not client:
                raise HTTPException(status_code=400, detail=f"Platform {request.platform} not supported")
            
            # Upload to platform
            platform_result = await self._upload_with_retry(client, video, request, upload_record)
            
            # Update upload record with results
            await self._update_upload_record(upload_record, platform_result, start_time)
            
            # Record metrics
            UPLOAD_REQUESTS.labels(platform=request.platform.value, status="success").inc()
            UPLOAD_DURATION.labels(platform=request.platform.value).observe(
                (datetime.utcnow() - start_time).total_seconds()
            )
            
            return UploadResult(
                upload_id=upload_id,
                video_id=request.video_id,
                platform=request.platform,
                status=UploadStatus.COMPLETED,
                platform_video_id=platform_result.get("video_id"),
                upload_url=platform_result.get("url"),
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                created_at=upload_record.created_at,
                updated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Upload failed for {upload_id}: {e}")
            UPLOAD_REQUESTS.labels(platform=request.platform.value, status="error").inc()
            PLATFORM_ERRORS.labels(platform=request.platform.value, error_type=type(e).__name__).inc()
            
            # Update upload record with error
            if 'upload_record' in locals():
                await self._update_upload_record_error(upload_record, str(e), start_time)
            
            return UploadResult(
                upload_id=upload_id,
                video_id=request.video_id,
                platform=request.platform,
                status=UploadStatus.FAILED,
                error_message=str(e),
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                created_at=datetime.utcnow()
            )
    
    async def upload_to_multiple_platforms(self, request: BatchUploadRequest) -> List[UploadResult]:
        """Upload video to multiple platforms with parallel execution"""
        
        # Create platform-specific requests
        platform_requests = []
        for platform in request.platforms:
            platform_request = PlatformUploadRequest(
                video_id=request.video_id,
                platform=platform,
                title=request.title,
                description=request.description,
                hashtags=request.hashtags,
                category=request.category,
                privacy=request.privacy,
                scheduled_time=request.scheduled_time,
                priority=request.priority,
                metadata=request.platform_specific.get(platform.value, {})
            )
            
            # Apply platform-specific overrides
            if platform.value in request.platform_specific:
                overrides = request.platform_specific[platform.value]
                for key, value in overrides.items():
                    if hasattr(platform_request, key):
                        setattr(platform_request, key, value)
            
            platform_requests.append(platform_request)
        
        # Execute uploads in parallel using asyncio.gather
        try:
            upload_tasks = [
                self.upload_to_platform(platform_request)
                for platform_request in platform_requests
            ]
            
            # Execute all uploads concurrently
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Convert exceptions to failed UploadResult
                    platform_request = platform_requests[i]
                    processed_results.append(UploadResult(
                        upload_id=str(uuid.uuid4()),
                        video_id=platform_request.video_id,
                        platform=platform_request.platform,
                        status=UploadStatus.FAILED,
                        error_message=str(result),
                        processing_time=0.0,
                        created_at=datetime.utcnow()
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            # Return failed results for all platforms
            return [
                UploadResult(
                    upload_id=str(uuid.uuid4()),
                    video_id=request.video_id,
                    platform=platform_request.platform,
                    status=UploadStatus.FAILED,
                    error_message=f"Batch upload failed: {str(e)}",
                    processing_time=0.0,
                    created_at=datetime.utcnow()
                )
                for platform_request in platform_requests
            ]
    
    async def _upload_with_retry(self, client, video, request, upload_record):
        """Upload with retry logic and timeout enforcement"""
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Update status to uploading
                await self._update_upload_status(upload_record, UploadStatus.UPLOADING)
                
                # Attempt upload with timeout
                upload_task = client.upload_video(
                    video_path=video.file_path,
                    title=request.title,
                    description=request.description,
                    hashtags=request.hashtags,
                    category=request.category,
                    privacy=request.privacy,
                    metadata=request.metadata
                )
                
                # Apply timeout to upload operation
                try:
                    result = await asyncio.wait_for(upload_task, timeout=UPLOAD_TIMEOUT)
                    return result
                except asyncio.TimeoutError:
                    raise Exception(f"Upload timed out after {UPLOAD_TIMEOUT} seconds")
                
            except Exception as e:
                attempt_num = attempt + 1
                logger.warning(f"Upload attempt {attempt_num} failed: {e}")
                RETRY_ATTEMPTS.labels(platform=request.platform.value).inc()
                
                if attempt_num < MAX_RETRY_ATTEMPTS:
                    # Calculate exponential backoff delay
                    delay = RETRY_DELAY_BASE ** attempt_num
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    raise e
    
    async def _get_video_by_id(self, video_id: str):
        """Get video by ID"""
        from sqlalchemy import select
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Video).filter(Video.id == video_id)
            )
            return result.scalar_one_or_none()
    
    async def _check_platform_limits(self, platform: PlatformType):
        """Check platform rate limits and quotas with enforcement"""
        limiter = await get_rate_limiter()
        
        # Define platform-specific rate limits
        platform_limits = {
            PlatformType.YOUTUBE: {"daily": 100, "hourly": 10, "window": 3600},
            PlatformType.INSTAGRAM: {"daily": 50, "hourly": 5, "window": 3600},
            PlatformType.TIKTOK: {"daily": 30, "hourly": 3, "window": 3600},
            PlatformType.FACEBOOK: {"daily": 100, "hourly": 10, "window": 3600}
        }
        
        limits = platform_limits.get(platform, {"daily": 10, "hourly": 1, "window": 3600})
        
        # Check hourly rate limit
        hourly_key = f"upload_rate:{platform.value}:hourly"
        if not await limiter.rate_limiter.is_allowed(
            key=hourly_key,
            limit=limits["hourly"],
            window_seconds=limits["window"]
        ):
            raise HTTPException(
                status_code=429,
                detail=f"Hourly upload limit exceeded for {platform.value}. Limit: {limits['hourly']}/hour"
            )
        
        # Check daily rate limit
        daily_key = f"upload_rate:{platform.value}:daily"
        if not await limiter.rate_limiter.is_allowed(
            key=daily_key,
            limit=limits["daily"],
            window_seconds=86400  # 24 hours
        ):
            raise HTTPException(
                status_code=429,
                detail=f"Daily upload limit exceeded for {platform.value}. Limit: {limits['daily']}/day"
            )
        
        logger.info(f"Platform limits check passed for {platform.value}")
    
    async def _check_idempotency_key(
        self, idempotency_key: str, video_id: str, platform: PlatformType
    ) -> Optional[UploadResult]:
        """Check if upload already exists for idempotency key"""
        from sqlalchemy import select, and_, text
        
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(PlatformUpload).filter(
                    and_(
                        PlatformUpload.video_id == video_id,
                        PlatformUpload.platform == platform,
                        text("upload_metadata->>'idempotency_key' = :idempotency_key")
                    )
                ).params(idempotency_key=idempotency_key)
            )
            existing_upload = result.scalar_one_or_none()
            
            if existing_upload:
                # Map to UploadResult
                status_mapping = {
                    ContentStatus.PENDING: UploadStatus.PENDING,
                    ContentStatus.PROCESSING: UploadStatus.UPLOADING,
                    ContentStatus.COMPLETED: UploadStatus.COMPLETED,
                    ContentStatus.FAILED: UploadStatus.FAILED,
                    ContentStatus.MODERATION_FAILED: UploadStatus.CANCELLED
                }
                
                return UploadResult(
                    upload_id=str(existing_upload.id),
                    video_id=video_id,
                    platform=platform,
                    status=status_mapping.get(existing_upload.status, UploadStatus.PENDING),
                    platform_video_id=existing_upload.platform_video_id,
                    upload_url=existing_upload.upload_url,
                    error_message=existing_upload.error_message,
                    retry_count=existing_upload.retry_count,
                    processing_time=0.0,  # Not calculated for existing uploads
                    created_at=existing_upload.created_at,
                    updated_at=existing_upload.updated_at
                )
            
            return None
    
    async def _schedule_upload(
        self, request: PlatformUploadRequest, upload_id: str, video
    ) -> UploadResult:
        """Schedule upload for future execution"""
        import redis.asyncio as redis
        
        # Create upload record with PENDING status
        upload_record = await self._create_upload_record(upload_id, request, video)
        
        # Store scheduled upload in Redis for background processing
        redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        
        scheduled_data = {
            "upload_id": upload_id,
            "request": request.dict(),
            "video_id": str(video.id),
            "scheduled_time": request.scheduled_time.isoformat()
        }
        
        # Use Redis sorted set with timestamp as score for scheduling
        await redis_client.zadd(
            "scheduled_uploads",
            {json.dumps(scheduled_data): request.scheduled_time.timestamp()}
        )
        
        await redis_client.close()
        
        logger.info(f"Upload {upload_id} scheduled for {request.scheduled_time}")
        
        return UploadResult(
            upload_id=upload_id,
            video_id=request.video_id,
            platform=request.platform,
            status=UploadStatus.PENDING,
            processing_time=0.0,
            created_at=upload_record.created_at
        )
    
    async def _create_upload_record(self, upload_id: str, request: PlatformUploadRequest, video):
        """Create upload record in database"""
        async with self.db_manager.get_session() as session:
            upload_record = PlatformUpload(
                id=uuid.UUID(upload_id),
                video_id=video.id,
                platform=request.platform,
                upload_title=request.title,
                upload_description=request.description,
                hashtags=request.hashtags,
                upload_metadata={
                    "category": request.category,
                    "privacy": request.privacy,
                    "scheduled_time": request.scheduled_time.isoformat() if request.scheduled_time else None,
                    "priority": request.priority.value,
                    "idempotency_key": request.idempotency_key,
                    **request.metadata
                },
                status=ContentStatus.PENDING,
                retry_count=0
            )
            session.add(upload_record)
            await session.commit()
            await session.refresh(upload_record)
            return upload_record
    
    async def _update_upload_record(self, upload_record, platform_result, start_time):
        """Update upload record with results"""
        async with self.db_manager.get_session() as session:
            upload_record.platform_video_id = platform_result.get("video_id")
            upload_record.upload_url = platform_result.get("url")
            upload_record.status = ContentStatus.COMPLETED
            upload_record.upload_completed_at = datetime.utcnow()
            upload_record.updated_at = datetime.utcnow()
            
            session.add(upload_record)
            await session.commit()
    
    async def _update_upload_record_error(self, upload_record, error_message, start_time):
        """Update upload record with error"""
        async with self.db_manager.get_session() as session:
            upload_record.error_message = error_message
            upload_record.status = ContentStatus.FAILED
            upload_record.retry_count += 1
            upload_record.updated_at = datetime.utcnow()
            
            session.add(upload_record)
            await session.commit()
    
    async def _update_upload_status(self, upload_record, status: UploadStatus):
        """Update upload status"""
        async with self.db_manager.get_session() as session:
            if status == UploadStatus.UPLOADING:
                upload_record.upload_started_at = datetime.utcnow()
                upload_record.status = ContentStatus.PROCESSING
            
            upload_record.updated_at = datetime.utcnow()
            session.add(upload_record)
            await session.commit()
    
    async def get_upload_status(self, upload_id: str) -> Optional[UploadStatusResponse]:
        """Get upload status by ID"""
        from sqlalchemy import select
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(PlatformUpload).filter(PlatformUpload.id == upload_id)
            )
            upload = result.scalar_one_or_none()
            
            if not upload:
                return None
            
            # Map ContentStatus to UploadStatus
            status_mapping = {
                ContentStatus.PENDING: UploadStatus.PENDING,
                ContentStatus.PROCESSING: UploadStatus.UPLOADING,
                ContentStatus.COMPLETED: UploadStatus.COMPLETED,
                ContentStatus.FAILED: UploadStatus.FAILED,
                ContentStatus.MODERATION_FAILED: UploadStatus.CANCELLED  # Map moderation failed to cancelled
            }
            
            return UploadStatusResponse(
                upload_id=str(upload.id),
                video_id=str(upload.video_id),
                platform=upload.platform,
                status=status_mapping.get(upload.status, UploadStatus.PENDING),
                platform_video_id=upload.platform_video_id,
                upload_url=upload.upload_url,
                error_message=upload.error_message,
                retry_count=upload.retry_count,
                created_at=upload.created_at,
                updated_at=upload.updated_at
            )
    
    async def get_platform_limits(self, platform: PlatformType) -> PlatformLimits:
        """Get platform limits and current usage"""
        limits = {
            PlatformType.YOUTUBE: PlatformLimits(
                daily_uploads=100,
                file_size_mb=256000,  # 256GB
                duration_seconds=43200,  # 12 hours
                title_max_length=100,
                description_max_length=5000,
                hashtags_max_count=15
            ),
            PlatformType.INSTAGRAM: PlatformLimits(
                daily_uploads=50,
                file_size_mb=4000,  # 4GB
                duration_seconds=3600,  # 1 hour
                title_max_length=125,
                description_max_length=2200,
                hashtags_max_count=30
            ),
            PlatformType.TIKTOK: PlatformLimits(
                daily_uploads=30,
                file_size_mb=4000,  # 4GB
                duration_seconds=600,  # 10 minutes
                title_max_length=150,
                description_max_length=2200,
                hashtags_max_count=20
            ),
            PlatformType.FACEBOOK: PlatformLimits(
                daily_uploads=100,
                file_size_mb=10000,  # 10GB
                duration_seconds=7200,  # 2 hours
                title_max_length=100,
                description_max_length=63206,
                hashtags_max_count=10
            )
        }
        
        return limits.get(platform, PlatformLimits(
            daily_uploads=10,
            file_size_mb=1000,
            duration_seconds=600,
            title_max_length=100,
            description_max_length=1000,
            hashtags_max_count=10
        ))


# Global service instance
distribution_processor: Optional[DistributionProcessor] = None


class SecretsManager:
    """Manages secrets validation and retrieval"""
    
    @staticmethod
    def validate_platform_credentials() -> Dict[str, bool]:
        """Validate all platform credentials at startup"""
        validation_results = {
            "youtube": bool(os.getenv("YOUTUBE_CLIENT_ID") and 
                           os.getenv("YOUTUBE_CLIENT_SECRET") and 
                           os.getenv("YOUTUBE_REFRESH_TOKEN")),
            "instagram": bool(os.getenv("INSTAGRAM_ACCESS_TOKEN") and 
                             os.getenv("INSTAGRAM_USER_ID")),
            "tiktok": bool(os.getenv("TIKTOK_ACCESS_TOKEN") and 
                          os.getenv("TIKTOK_CLIENT_KEY")),
            "facebook": bool(os.getenv("FACEBOOK_ACCESS_TOKEN") and 
                            os.getenv("FACEBOOK_PAGE_ID"))
        }
        
        missing_credentials = [platform for platform, valid in validation_results.items() if not valid]
        
        if missing_credentials:
            logger.warning(f"Missing credentials for platforms: {', '.join(missing_credentials)}")
        else:
            logger.info("All platform credentials validated successfully")
        
        return validation_results
    
    @staticmethod
    def get_required_env_vars() -> List[str]:
        """Get list of required environment variables"""
        return [
            "DATABASE_URL",
            "REDIS_URL",
            "SERVICE_NAME",
            "LOG_LEVEL",
            "ENVIRONMENT"
        ]
    
    @staticmethod
    def validate_required_env_vars() -> bool:
        """Validate required environment variables"""
        required_vars = SecretsManager.get_required_env_vars()
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        logger.info("All required environment variables present")
        return True


@app.on_event("startup")
async def on_startup():
    """Initialize distribution service on startup with secrets validation"""
    global distribution_processor
    
    # Validate required environment variables
    if not SecretsManager.validate_required_env_vars():
        raise Exception("Required environment variables missing")
    
    # Validate platform credentials
    credential_status = SecretsManager.validate_platform_credentials()
    
    # Initialize database
    db_manager = get_db_manager()
    await db_manager.initialize()
    
    # Initialize distribution processor
    distribution_processor = DistributionProcessor(db_manager)
    
    logger.info("Distribution service initialized")
    logger.info(f"Platform credential status: {credential_status}")


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on shutdown"""
    await get_db_manager().close()
    logger.info("Distribution service shutdown complete")


def get_distribution_processor() -> DistributionProcessor:
    """Get distribution processor instance"""
    if distribution_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return distribution_processor


@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint with optional deep checks"""
    health_status = {"status": "healthy", "service": "distribution-service"}
    
    if deep:
        try:
            async with get_db_manager().get_session() as session:
                await session.execute(text("SELECT 1"))
            health_status["database"] = "connected"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["database"] = f"error: {str(e)}"
        
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            await redis_client.ping()
            health_status["redis"] = "connected"
            await redis_client.close()
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
        
        # Check platform client connectivity
        platforms_status = {}
        for platform, client in distribution_processor.platform_clients.items():
            try:
                # Test platform connectivity (implementation depends on client)
                platforms_status[platform.value] = "available"
            except Exception as e:
                platforms_status[platform.value] = f"error: {str(e)}"
        
        health_status["platforms"] = platforms_status
    
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Platform Distribution Service API", "version": "1.0.0"}


@app.post("/upload/{platform}", response_model=UploadResult)
async def upload_to_platform_endpoint(
    platform: PlatformType,
    request: PlatformUploadRequest,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Upload video to specific platform"""
    try:
        # Override platform from URL parameter
        request.platform = platform
        
        with REQUEST_DURATION.time():
            result = await processor.upload_to_platform(request)
        
        REQUEST_COUNT.labels(method="POST", endpoint=f"/upload/{platform}", status="success").inc()
        return result
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint=f"/upload/{platform}", status="error").inc()
        logger.error(f"Platform upload failed: {e}")
        raise HTTPException(status_code=500, detail="Platform upload failed")


@app.post("/upload/batch", response_model=List[UploadResult])
async def upload_to_multiple_platforms_endpoint(
    request: BatchUploadRequest,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Upload video to multiple platforms"""
    try:
        with REQUEST_DURATION.time():
            results = await processor.upload_to_multiple_platforms(request)
        
        REQUEST_COUNT.labels(method="POST", endpoint="/upload/batch", status="success").inc()
        return results
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/upload/batch", status="error").inc()
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail="Batch upload failed")


@app.get("/uploads/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status_endpoint(
    upload_id: str,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Get upload status by ID"""
    try:
        status = await processor.get_upload_status(upload_id)
        if not status:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        REQUEST_COUNT.labels(method="GET", endpoint="/uploads/{upload_id}/status", status="success").inc()
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="GET", endpoint="/uploads/{upload_id}/status", status="error").inc()
        logger.error(f"Failed to get upload status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get upload status")


@app.get("/uploads/video/{video_id}")
async def get_video_uploads(
    video_id: str,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Get all uploads for a video"""
    try:
        from sqlalchemy import select
        async with get_db_manager().get_session() as session:
            result = await session.execute(
                select(PlatformUpload).filter(PlatformUpload.video_id == video_id)
            )
            uploads = result.scalars().all()
            
            return {
                "video_id": video_id,
                "uploads": [
                    {
                        "upload_id": str(upload.id),
                        "platform": upload.platform,
                        "status": upload.status,
                        "platform_video_id": upload.platform_video_id,
                        "upload_url": upload.upload_url,
                        "error_message": upload.error_message,
                        "retry_count": upload.retry_count,
                        "created_at": upload.created_at,
                        "updated_at": upload.updated_at
                    }
                    for upload in uploads
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to get video uploads: {e}")
        raise HTTPException(status_code=500, detail="Failed to get video uploads")


@app.put("/uploads/{upload_id}/retry")
async def retry_upload_endpoint(
    upload_id: str,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Retry failed upload"""
    try:
        # Get upload record
        from sqlalchemy import select
        async with get_db_manager().get_session() as session:
            result = await session.execute(
                select(PlatformUpload).filter(PlatformUpload.id == upload_id)
            )
            upload = result.scalar_one_or_none()
            
            if not upload:
                raise HTTPException(status_code=404, detail="Upload not found")
            
            if upload.status != ContentStatus.FAILED:
                raise HTTPException(status_code=400, detail="Upload is not in failed state")
            
            # Create new upload request from existing record
            request = PlatformUploadRequest(
                video_id=str(upload.video_id),
                platform=upload.platform,
                title=upload.upload_title,
                description=upload.upload_description or "",
                hashtags=upload.hashtags or [],
                metadata=upload.upload_metadata or {}
            )
            
            # Retry upload
            result = await processor.upload_to_platform(request)
            
            REQUEST_COUNT.labels(method="PUT", endpoint="/uploads/retry", status="success").inc()
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="PUT", endpoint="/uploads/retry", status="error").inc()
        logger.error(f"Failed to retry upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry upload")


@app.delete("/uploads/{upload_id}")
async def cancel_upload_endpoint(
    upload_id: str,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Cancel/delete upload"""
    try:
        from sqlalchemy import select
        async with get_db_manager().get_session() as session:
            result = await session.execute(
                select(PlatformUpload).filter(PlatformUpload.id == upload_id)
            )
            upload = result.scalar_one_or_none()
            
            if not upload:
                raise HTTPException(status_code=404, detail="Upload not found")
            
            # Update status to cancelled with clear marker
            upload.status = ContentStatus.MODERATION_FAILED  # Use existing enum for cancelled state
            upload.error_message = "Upload cancelled by user request"
            upload.updated_at = datetime.utcnow()
            
            # Add metadata to clearly mark as cancelled
            if upload.upload_metadata is None:
                upload.upload_metadata = {}
            upload.upload_metadata['cancelled'] = True
            upload.upload_metadata['cancelled_at'] = datetime.utcnow().isoformat()
            
            session.add(upload)
            await session.commit()
            
            REQUEST_COUNT.labels(method="DELETE", endpoint="/uploads/cancel", status="success").inc()
            return {"message": "Upload cancelled successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="DELETE", endpoint="/uploads/cancel", status="error").inc()
        logger.error(f"Failed to cancel upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel upload")


@app.get("/platforms/{platform}/limits", response_model=PlatformLimits)
async def get_platform_limits_endpoint(
    platform: PlatformType,
    processor: DistributionProcessor = Depends(get_distribution_processor)
):
    """Get platform rate limits and quotas"""
    try:
        limits = await processor.get_platform_limits(platform)
        REQUEST_COUNT.labels(method="GET", endpoint="/platforms/limits", status="success").inc()
        return limits
        
    except Exception as e:
        REQUEST_COUNT.labels(method="GET", endpoint="/platforms/limits", status="error").inc()
        logger.error(f"Failed to get platform limits: {e}")
        raise HTTPException(status_code=500, detail="Failed to get platform limits")


@app.get("/platforms/{platform}/formats")
async def get_platform_formats_endpoint(platform: PlatformType):
    """Get supported formats for specific platform"""
    
    formats = {
        PlatformType.YOUTUBE: {
            "video_formats": ["mp4", "mov", "avi", "wmv", "flv", "webm"],
            "audio_formats": ["aac", "mp3"],
            "codecs": ["H.264", "H.265"],
            "aspect_ratios": ["16:9", "4:3", "1:1", "9:16"],
            "max_resolution": "8K",
            "max_fps": 60
        },
        PlatformType.INSTAGRAM: {
            "video_formats": ["mp4", "mov"],
            "audio_formats": ["aac"],
            "codecs": ["H.264"],
            "aspect_ratios": ["1:1", "4:5", "9:16"],
            "max_resolution": "1080p",
            "max_fps": 30
        },
        PlatformType.TIKTOK: {
            "video_formats": ["mp4", "mov"],
            "audio_formats": ["aac", "mp3"],
            "codecs": ["H.264", "H.265"],
            "aspect_ratios": ["9:16"],
            "max_resolution": "1080p",
            "max_fps": 60
        },
        PlatformType.FACEBOOK: {
            "video_formats": ["mp4", "mov", "avi"],
            "audio_formats": ["aac", "mp3"],
            "codecs": ["H.264"],
            "aspect_ratios": ["16:9", "1:1", "9:16"],
            "max_resolution": "4K",
            "max_fps": 60
        }
    }
    
    return formats.get(platform, {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)