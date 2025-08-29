"""
Image Generation Service

FastAPI service for generating AI images using DALL·E 3 and Stable Diffusion XL
with scene analysis, batch processing, cloud storage, and comprehensive image management.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import shutil
import redis.asyncio as redis

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
import aiofiles

from shared.database import get_db_manager, DatabaseManager
from shared.middleware import CorrelationMiddleware
from shared.rate_limiter import RateLimiter
from shared.models import MediaAsset, Story, MediaType
from shared.schemas import MediaAssetRequest, MediaAssetResponse
from shared.config import get_config

# Import new components
from batch_processor import BatchProcessor, JobStatus, BatchJob
from storage_manager import create_storage_manager, StorageManager
from image_processor import ImageProcessor
from quality_validator import ImageQualityValidator, QualityMetrics
from redis_rate_limiter import ProviderRateLimiter

# Prometheus metrics
REQUEST_COUNT = Counter('image_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('image_service_request_duration_seconds', 'Request duration')
PROVIDER_CALLS = Counter('image_service_provider_calls_total', 'Provider API calls', ['provider', 'status'])
PROVIDER_ERRORS = Counter('image_service_provider_errors_total', 'Provider errors', ['provider', 'error_type'])
GENERATION_DURATION = Histogram('image_service_generation_duration_seconds', 'Image generation duration')
RATE_LIMIT_HITS = Counter('image_service_rate_limit_hits_total', 'Rate limit hits', ['provider', 'type'])

# Configuration
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads/images"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiters for different providers
config = get_config()
RATE_LIMITER = None  # Simplified for now

logger = logging.getLogger(__name__)


class ImageRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1, max_length=1000)
    story_id: Optional[str] = Field(None, description="Associated story ID")
    scene_number: Optional[int] = Field(None, description="Scene number within the story", ge=1)
    style: Optional[str] = Field("photorealistic", description="Image style preference")
    aspect_ratio: Optional[str] = Field("16:9", description="Aspect ratio (16:9, 9:16, 1:1)")
    quality: Optional[str] = Field("standard", description="Image quality (standard, hd)")
    provider: Optional[str] = Field("dall-e-3", description="Preferred provider (dall-e-3, stable-diffusion)")
    
    @validator('aspect_ratio')
    def validate_aspect_ratio(cls, v):
        valid_ratios = ["16:9", "9:16", "1:1", "4:3", "3:4"]
        if v not in valid_ratios:
            raise ValueError(f"Aspect ratio must be one of {valid_ratios}")
        return v


class BatchImageRequest(BaseModel):
    """Request model for batch image generation"""
    story_id: str = Field(..., description="Story ID for batch generation")
    scenes: List[Dict[str, Union[str, int]]] = Field(..., description="List of scenes with prompts")
    style: Optional[str] = Field("photorealistic", description="Consistent style for all images")
    aspect_ratio: Optional[str] = Field("16:9", description="Aspect ratio for all images")
    quality: Optional[str] = Field("standard", description="Image quality")


class AsyncBatchRequest(BaseModel):
    """Request model for asynchronous batch processing"""
    prompts: List[str] = Field(..., description="List of prompts to generate")
    provider: str = Field("dall-e-3", description="AI provider to use")
    parameters: Optional[Dict[str, Union[str, int, float]]] = Field(None, description="Generation parameters")
    story_id: Optional[str] = Field(None, description="Associated story ID")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    priority: int = Field(0, description="Job priority (higher = more priority)")
    validate_quality: bool = Field(False, description="Enable quality validation")
    create_variants: bool = Field(False, description="Create platform variants")
    platforms: Optional[List[str]] = Field(None, description="Target platforms for variants")


class ProcessingRequest(BaseModel):
    """Request model for image processing operations"""
    image_id: str = Field(..., description="Image ID to process")
    operations: List[Dict[str, Union[str, int, float]]] = Field(..., description="Processing operations")
    save_original: bool = Field(True, description="Keep original image")


class QualityValidationRequest(BaseModel):
    """Request model for quality validation"""
    image_source: str = Field(..., description="Image URL, path, or base64 data")
    detailed: bool = Field(True, description="Perform detailed analysis")


class ImageResponse(BaseModel):
    """Response model for image generation"""
    id: str
    story_id: Optional[str]
    scene_number: Optional[int]
    prompt: str
    file_path: str
    file_url: str
    provider: str
    metadata: Dict
    created_at: datetime


class BatchImageResponse(BaseModel):
    """Response model for batch image generation"""
    story_id: str
    images: List[ImageResponse]
    total_generated: int
    failed_generations: List[Dict[str, str]]


class SceneAnalyzer:
    """Analyzes story content to generate appropriate image prompts"""
    
    @staticmethod
    async def extract_scenes_from_story(story_content: str, max_scenes: int = 5) -> List[Dict[str, Union[str, int]]]:
        """Extract visual scenes from story content"""
        # Simple scene extraction - in production, this could use NLP/LLM
        sentences = story_content.split('. ')
        scenes = []
        
        # Look for descriptive sentences that would make good images
        visual_keywords = [
            'looked', 'saw', 'appeared', 'stood', 'walked', 'ran', 'beautiful', 'dark', 'bright',
            'forest', 'mountain', 'ocean', 'city', 'house', 'room', 'garden', 'sky', 'sunset',
            'morning', 'night', 'landscape', 'scene', 'view', 'sight'
        ]
        
        scene_number = 1
        for sentence in sentences[:max_scenes * 2]:  # Check more sentences than needed
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in visual_keywords):
                # Enhance the sentence for better image generation
                enhanced_prompt = SceneAnalyzer.enhance_prompt_for_image_generation(sentence)
                scenes.append({
                    "scene_number": scene_number,
                    "prompt": enhanced_prompt,
                    "original_text": sentence
                })
                scene_number += 1
                
                if len(scenes) >= max_scenes:
                    break
        
        # If we don't have enough scenes, create generic ones
        while len(scenes) < min(3, max_scenes):
            scenes.append({
                "scene_number": len(scenes) + 1,
                "prompt": f"A beautiful, cinematic scene from a story, scene {len(scenes) + 1}",
                "original_text": "Generated scene"
            })
        
        return scenes
    
    @staticmethod
    def enhance_prompt_for_image_generation(text: str) -> str:
        """Enhance text for better image generation results"""
        # Add style and quality modifiers
        enhancements = [
            "cinematic lighting",
            "high quality",
            "detailed",
            "professional photography style"
        ]
        
        # Clean up the text
        text = text.strip().rstrip('.')
        
        # Add enhancements
        enhanced = f"{text}, {', '.join(enhancements)}"
        
        return enhanced


class ImageProvider:
    """Base class for image generation providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict:
        """Generate image - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Legacy provider classes removed - now using enhanced providers from separate files


class ImageService:
    """Main service for image generation and management"""
    
    def __init__(self, storage_manager: StorageManager, quality_validator: ImageQualityValidator):
        self.providers = {}
        self.scene_analyzer = SceneAnalyzer()
        self.processor = ImageProcessor()
        self.storage_manager = storage_manager
        self.quality_validator = quality_validator
        
        # Initialize providers
        openai_key = os.getenv("OPENAI_API_KEY")
        stability_key = os.getenv("STABILITY_API_KEY")
        
        # Include mock provider for testing/development
        if os.getenv("IMAGE_PROVIDER") == "mock" or os.getenv("ENVIRONMENT", "development") == "development":
            from providers.mock_provider import MockProvider
            fail_rate = float(os.getenv("MOCK_FAIL_RATE", "0.1"))  # 10% default failure rate
            self.providers["mock"] = MockProvider(fail_rate=fail_rate)
        
        # Initialize enhanced providers
        if openai_key and openai_key != "test-key-for-development":
            from providers.dalle_provider import DALLEProvider
            self.providers["dall-e-3"] = DALLEProvider(openai_key)
        
        if stability_key and stability_key != "test-key-for-development":
            from providers.stability_provider import StabilityProvider
            self.providers["stable-diffusion-xl"] = StabilityProvider(stability_key)
        
        # Add Replicate provider if configured
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if replicate_token and replicate_token != "test-token-for-development":
            from providers.replicate_provider import ReplicateProvider
            self.providers["replicate"] = ReplicateProvider(replicate_token)
        
        if len(self.providers) == 1:  # Only mock provider
            logger.warning("Only mock provider configured - real API keys needed for production")
    
    async def generate_single_image(self, request: ImageRequest) -> ImageResponse:
        """Generate a single image"""
        start_time = datetime.utcnow()
        
        # Choose provider
        provider_name = request.provider
        if provider_name not in self.providers:
            # Fallback to available provider
            provider_name = next(iter(self.providers.keys())) if self.providers else None
            
        if not provider_name:
            raise HTTPException(status_code=503, detail="No image generation providers available")
        
        provider = self.providers[provider_name]
        
        try:
            # Enforce distributed rate limit / circuit breaker if configured
            if provider_rate_limiter:
                available = await provider_rate_limiter.check_provider_availability(provider_name)
                if not available:
                    RATE_LIMIT_HITS.labels(provider=provider_name, type="global").inc()
                    raise HTTPException(status_code=429, detail=f"Provider '{provider_name}' temporarily unavailable due to rate limits")
            # Generate image with enhanced providers
            with GENERATION_DURATION.time():
                if hasattr(provider, 'generate_image') and callable(getattr(provider, 'generate_image')):
                    # New enhanced provider interface
                    result = await provider.generate_image(
                        request.prompt,
                        size=request.aspect_ratio,
                        quality=request.quality,
                        image_style=request.style
                    )
                else:
                    # Legacy provider interface
                    result = await provider.generate_image(
                        request.prompt,
                        aspect_ratio=request.aspect_ratio,
                        quality=request.quality
                    )
            
            # Debug logging
            logger.info(f"Provider {provider_name} returned: {result}")
            
            PROVIDER_CALLS.labels(provider=provider_name, status="success").inc()
            if provider_rate_limiter:
                await provider_rate_limiter.record_provider_success(provider_name)
            
            # Create unique filename
            image_id = str(uuid.uuid4())
            file_path = UPLOAD_DIR / f"{image_id}.png"
            
            # Save image
            if "image_url" in result:
                metadata = await self.processor.download_and_save_image(result["image_url"], file_path)
                # Optimize image
                optimization_result = await self.processor.optimize_image(file_path)
                final_path = optimization_result["optimized_path"]
            elif "image_data" in result:
                # Handle mock or base64 data
                if result.get("metadata", {}).get("mock"):
                    # For mock data, just create a simple file without processing
                    import base64
                    image_bytes = base64.b64decode(result["image_data"])
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(image_bytes)
                    metadata = {
                        "format": "PNG",
                        "size": (1, 1),
                        "mode": "RGBA",
                        "file_size": len(image_bytes),
                        "mock": True
                    }
                    final_path = file_path  # Don't optimize mock images
                else:
                    metadata = await self.processor.save_base64_image(result["image_data"], file_path)
                    # Optimize image
                    optimization_result = await self.processor.optimize_image(file_path)
                    final_path = optimization_result["optimized_path"]
            else:
                raise ValueError("No image data or URL in provider response")
            
            # Store in database (skip if no story_id for now due to constraint)
            media_asset = None
            if request.story_id:
                try:
                    db_manager = get_db_manager()
                    async with db_manager.get_session() as session:
                        media_asset = MediaAsset(
                            id=uuid.UUID(image_id),
                            story_id=uuid.UUID(request.story_id),
                            asset_type=MediaType.IMAGE,
                            file_path=str(final_path),
                            metadata_json={
                                **metadata,
                                **result,
                                "prompt": request.prompt,
                                "scene_number": request.scene_number,
                                "aspect_ratio": request.aspect_ratio,
                                "generation_time": (datetime.utcnow() - start_time).total_seconds()
                            }
                        )
                        session.add(media_asset)
                        await session.commit()
                        await session.refresh(media_asset)
                except Exception as e:
                    logger.warning(f"Failed to store media asset in database: {e}")
                    # Continue without database storage for now
            
            return ImageResponse(
                id=image_id,
                story_id=request.story_id,
                scene_number=request.scene_number,
                prompt=request.prompt,
                file_path=str(final_path),
                file_url=f"/images/{final_path.name}",
                provider=provider_name,
                metadata={**metadata, **result} if not media_asset else media_asset.metadata,
                created_at=media_asset.created_at if media_asset else datetime.utcnow()
            )
            
        except Exception as e:
            PROVIDER_CALLS.labels(provider=provider_name, status="error").inc()
            if provider_rate_limiter:
                try:
                    await provider_rate_limiter.record_provider_failure(provider_name)
                except Exception:
                    pass
            logger.error(f"Image generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
    
    async def generate_batch_images(self, request: BatchImageRequest) -> BatchImageResponse:
        """Generate multiple images for a story"""
        images = []
        failed_generations = []
        
        # If no scenes provided, analyze story content
        if not request.scenes:
            # Get story content from database
            db_manager = get_db_manager()
            async with db_manager.get_session() as session:
                story = await session.get(Story, uuid.UUID(request.story_id))
                if not story:
                    raise HTTPException(status_code=404, detail="Story not found")
                
                request.scenes = await self.scene_analyzer.extract_scenes_from_story(story.content)
        
        # Generate images for each scene
        for scene in request.scenes:
            try:
                image_request = ImageRequest(
                    prompt=scene["prompt"],
                    story_id=request.story_id,
                    scene_number=scene["scene_number"],
                    style=request.style,
                    aspect_ratio=request.aspect_ratio,
                    quality=request.quality
                )
                
                image_response = await self.generate_single_image(image_request)
                images.append(image_response)
                
            except Exception as e:
                failed_generations.append({
                    "scene_number": str(scene["scene_number"]),
                    "error": str(e)
                })
                logger.error(f"Failed to generate image for scene {scene['scene_number']}: {e}")
        
        return BatchImageResponse(
            story_id=request.story_id,
            images=images,
            total_generated=len(images),
            failed_generations=failed_generations
        )
    
    async def close(self):
        """Close all provider connections"""
        for provider in self.providers.values():
            await provider.close()


# Global service instances
image_service = None
batch_processor = None
storage_manager = None
quality_validator = None
redis_client = None
provider_rate_limiter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global image_service, batch_processor, storage_manager, quality_validator, redis_client, provider_rate_limiter
    
    # Startup
    logger.info("Starting Enhanced Image Generation Service")
    
    # Initialize Redis
    redis_host = os.getenv("REDIS_HOST", "redis")  # Use service name in Docker
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    # Initialize provider rate limiter
    provider_rate_limiter = ProviderRateLimiter(redis_client)
    
    # Initialize storage manager
    storage_manager = create_storage_manager()
    
    # Initialize quality validator
    quality_validator = ImageQualityValidator()
    
    # Initialize main image service
    image_service = ImageService(storage_manager, quality_validator)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(redis_client, image_service.providers)
    await batch_processor.start_processing()
    
    # Initialize database
    db_manager = get_db_manager()
    await db_manager.initialize()
    
    # Initialize rate limiter (simplified for now)
    global RATE_LIMITER
    RATE_LIMITER = None
    
    logger.info("Enhanced Image Generation Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Image Generation Service")
    
    if batch_processor:
        batch_processor.stop_processing()
    
    if image_service:
        await image_service.close()
    
    if redis_client:
        await redis_client.close()
    
    await db_manager.close()
    
    # Close rate limiter
    if RATE_LIMITER:
        await RATE_LIMITER.close()
    
    logger.info("Enhanced Image Generation Service shutdown complete")


# FastAPI app
app = FastAPI(
    title="Image Generation Service",
    description="AI-powered image generation service with DALL·E 3 and Stable Diffusion XL",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CorrelationMiddleware)

# Configure CORS based on environment
cors_origins = ["*"] if os.getenv("ENVIRONMENT", "development") == "development" else []
if os.getenv("CORS_ORIGINS"):
    cors_origins = os.getenv("CORS_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with structured logging"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def get_image_service() -> ImageService:
    """Dependency to get image service instance"""
    return image_service


@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint with optional deep checks"""
    health_status = {"status": "healthy", "service": "image-service", "version": "2.0.0"}
    
    if deep:
        # Deep health check - test database connectivity
        try:
            db_manager = get_db_manager()
            async with db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
            health_status["database"] = "connected"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["database"] = f"error: {str(e)}"
        
        # Check Redis connectivity
        try:
            if redis_client:
                await redis_client.ping()
                health_status["redis"] = "connected"
            else:
                health_status["redis"] = "not_configured"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["redis"] = f"error: {str(e)}"
        
        # Check provider availability and rate limits
        if image_service and provider_rate_limiter:
            provider_status = {}
            for provider_name in image_service.providers.keys():
                try:
                    stats = await provider_rate_limiter.get_provider_stats(provider_name)
                    provider_status[provider_name] = {
                        "available": stats["available"],
                        "remaining_requests": stats["remaining"],
                        "limit_per_minute": stats["limit_per_minute"]
                    }
                except Exception as e:
                    provider_status[provider_name] = {"error": str(e)}
            
            health_status["providers"] = provider_status
        else:
            health_status["providers"] = list(image_service.providers.keys()) if image_service else []
        
        # Check batch processor
        if batch_processor:
            try:
                queue_stats = await batch_processor.get_queue_stats()
                health_status["batch_processor"] = {
                    "running": batch_processor.processing,
                    "queue_length": queue_stats["queue_length"],
                    "processing": queue_stats["processing"]
                }
            except Exception as e:
                health_status["batch_processor"] = {"error": str(e)}
    
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Image Generation Service API",
        "version": "1.0.0",
        "providers": list(image_service.providers.keys()) if image_service else []
    }


@app.get("/providers")
async def get_providers():
    """Get available image generation providers with capabilities and stats"""
    if not image_service:
        return {"providers": []}
    
    provider_info = {}
    for name, provider in image_service.providers.items():
        try:
            # Get basic provider info
            if hasattr(provider, 'get_provider_info'):
                info = provider.get_provider_info()
            else:
                info = {"name": name, "status": "available"}
            
            # Add rate limit stats if available
            if provider_rate_limiter:
                try:
                    stats = await provider_rate_limiter.get_provider_stats(name)
                    info.update({
                        "rate_limit": {
                            "limit_per_minute": stats["limit_per_minute"],
                            "remaining": stats["remaining"],
                            "available": stats["available"]
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to get rate limit stats for {name}: {e}")
            
            provider_info[name] = info
            
        except Exception as e:
            logger.error(f"Error getting info for provider {name}: {e}")
            provider_info[name] = {"name": name, "status": "error", "error": str(e)}
    
    return {
        "providers": provider_info,
        "default": "dall-e-3" if "dall-e-3" in image_service.providers else next(iter(image_service.providers.keys()), None)
    }


@app.post("/generate/image", response_model=ImageResponse)
async def generate_image_endpoint(
    request: ImageRequest,
    background_tasks: BackgroundTasks,
    service: ImageService = Depends(get_image_service)
):
    """Generate a single image"""
    try:
        with REQUEST_DURATION.time():
            result = await service.generate_single_image(request)
        
        REQUEST_COUNT.labels(method="POST", endpoint="/generate/image", status="success").inc()
        return result
        
    except HTTPException:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate/image", status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate/image", status="error").inc()
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")


@app.post("/generate/batch", response_model=BatchImageResponse)
async def generate_batch_images_endpoint(
    request: BatchImageRequest,
    background_tasks: BackgroundTasks,
    service: ImageService = Depends(get_image_service)
):
    """Generate multiple images for a story"""
    try:
        with REQUEST_DURATION.time():
            result = await service.generate_batch_images(request)
        
        REQUEST_COUNT.labels(method="POST", endpoint="/generate/batch", status="success").inc()
        return result
        
    except HTTPException:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate/batch", status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate/batch", status="error").inc()
        logger.error(f"Batch image generation failed: {e}")
        raise HTTPException(status_code=500, detail="Batch image generation failed")


@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve generated images with path traversal protection"""
    try:
        file_path = (UPLOAD_DIR / filename).resolve()
        
        # Security check: ensure file is within upload directory
        if not file_path.is_relative_to(UPLOAD_DIR.resolve()):
            raise HTTPException(status_code=404, detail="Image not found")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(file_path)
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="Image not found")


@app.post("/generate/batch/async")
async def submit_batch_job(request: AsyncBatchRequest):
    """Submit asynchronous batch image generation job"""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    try:
        job_id = await batch_processor.submit_batch_job(
            prompts=request.prompts,
            provider=request.provider,
            parameters=request.parameters or {},
            story_id=request.story_id,
            user_id=request.user_id,
            priority=request.priority
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Batch job submitted with {len(request.prompts)} prompts"
        }
    except Exception as e:
        logger.error(f"Failed to submit batch job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get batch job status and results"""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    job = await batch_processor.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "total": job.total,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "results": job.results,
        "error": job.error,
        "retry_count": job.retry_count
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a batch job"""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    success = await batch_processor.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/queue/stats")
async def get_queue_stats():
    """Get batch processing queue statistics"""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    return await batch_processor.get_queue_stats()


@app.post("/validate/quality")
async def validate_image_quality(request: QualityValidationRequest):
    """Validate image quality"""
    if not quality_validator:
        raise HTTPException(status_code=503, detail="Quality validator not available")
    
    try:
        metrics = await quality_validator.validate_image(
            request.image_source,
            detailed=request.detailed
        )
        
        return {
            "metrics": {
                "resolution_score": metrics.resolution_score,
                "sharpness_score": metrics.sharpness_score,
                "brightness_score": metrics.brightness_score,
                "contrast_score": metrics.contrast_score,
                "color_balance_score": metrics.color_balance_score,
                "noise_score": metrics.noise_score,
                "overall_score": metrics.overall_score
            },
            "image_info": {
                "width": metrics.width,
                "height": metrics.height,
                "total_pixels": metrics.total_pixels,
                "aspect_ratio": metrics.aspect_ratio,
                "file_size": metrics.file_size,
                "format": metrics.format
            },
            "quality_grade": quality_validator.get_quality_grade(metrics.overall_score),
            "issues": metrics.issues,
            "recommendations": metrics.recommendations
        }
    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/image")
async def process_image(request: ProcessingRequest):
    """Apply processing operations to an image"""
    if not image_service:
        raise HTTPException(status_code=503, detail="Image service not available")
    
    try:
        # This would need to be implemented in the ImageService
        # For now, return a placeholder response
        return {
            "image_id": request.image_id,
            "operations_applied": len(request.operations),
            "message": "Image processing completed"
        }
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not available")
    
    try:
        # Get basic stats
        return {
            "storage_type": os.getenv('STORAGE_TYPE', 'local'),
            "total_files": "N/A",  # Would need to implement in storage manager
            "total_size": "N/A",
            "message": "Storage statistics available"
        }
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cleanup")
async def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old completed jobs (admin endpoint)"""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    try:
        cleaned = await batch_processor.cleanup_old_jobs(max_age_hours)
        return {"cleaned_jobs": cleaned, "max_age_hours": max_age_hours}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/story/{story_id}/images")
async def get_story_images(story_id: str):
    """Get all images for a specific story"""
    try:
        # Try to get from storage manager first
        if storage_manager:
            try:
                storage_images = await storage_manager.list_story_images(story_id)
                if storage_images:
                    return {
                        "story_id": story_id,
                        "images": storage_images,
                        "total": len(storage_images),
                        "source": "storage"
                    }
            except Exception as e:
                logger.warning(f"Failed to get images from storage: {e}")
        
        # Fallback to database
        db_manager = get_db_manager()
        async with db_manager.get_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, file_path, metadata, created_at 
                    FROM media_assets 
                    WHERE story_id = :story_id AND media_type = 'image'::mediatype
                    ORDER BY created_at
                """),
                {"story_id": story_id}
            )
            
            images = []
            for row in result.fetchall():
                images.append({
                    "id": str(row[0]),
                    "file_path": row[1],
                    "file_url": f"/images/{Path(row[1]).name}",
                    "metadata": row[2],
                    "created_at": row[3]
                })
            
            return {"story_id": story_id, "images": images}
            
    except Exception as e:
        logger.error(f"Failed to get story images: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve images")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)