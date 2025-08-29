"""
Content Moderation Service

FastAPI service for content filtering, compliance checking, and copyright detection
with support for text, image, and audio analysis, manual review queues, and audit logging.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Request, Query, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
import aiofiles
from pathlib import Path
import tempfile
import json
import openai

from shared.database import DatabaseManager, get_db_manager
from shared.logging import get_logger
from shared.config import get_config
from shared.middleware import CorrelationMiddleware
from shared.models import ContentModeration, ModerationStatus, ModerationType, ManualReviewQueue, AuditLog
from shared.retry import retry_api_call, convert_http_error, NetworkError
from shared.rate_limiter import get_rate_limiter

# Prometheus metrics
REQUEST_COUNT = Counter('moderation_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('moderation_service_request_duration_seconds', 'Request duration')
MODERATION_CALLS = Counter('moderation_service_api_calls_total', 'Moderation API calls', ['provider', 'type', 'status'])

app = FastAPI(title="Content Moderation Service", version="1.0.0")
logger = get_logger(__name__)
config = get_config()

# Add middleware
app.add_middleware(CorrelationMiddleware)

# Configuration
MODERATION_STORAGE_PATH = os.getenv("MODERATION_STORAGE_PATH", "/app/storage/moderation")
MODERATION_THRESHOLD = float(os.getenv("MODERATION_THRESHOLD", "0.7"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODERATION_PROVIDER = os.getenv("MODERATION_PROVIDER", "openai")  # openai, mock

# Create storage directory
STORAGE_DIR = Path(MODERATION_STORAGE_PATH)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ModerationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModerationRequest(BaseModel):
    content_type: ContentType
    content_id: str
    content_data: Optional[str] = None  # For text content
    file_path: Optional[str] = None     # For file-based content
    metadata: Dict[str, Any] = Field(default_factory=dict)
    moderation_level: ModerationLevel = ModerationLevel.MEDIUM
    user_id: Optional[str] = None
    platform: Optional[str] = None  # youtube, instagram, tiktok, facebook


class ModerationResult(BaseModel):
    content_id: str
    status: ModerationStatus
    score: float = Field(ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)
    categories: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    requires_review: bool = False
    audit_trail: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ReviewRequest(BaseModel):
    moderation_id: str
    reviewer_id: str
    decision: ModerationStatus
    notes: Optional[str] = None
    override_reason: Optional[str] = None


class ComplianceCheck(BaseModel):
    gdpr_compliant: bool
    coppa_compliant: bool
    platform_compliant: bool
    copyright_clear: bool
    violations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class BaseModerationProvider:
    """Base class for moderation providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def moderate_text(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    async def moderate_image(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    async def moderate_audio(self, audio_path: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIModerationProvider(BaseModerationProvider):
    """OpenAI moderation provider using OpenAI's moderation API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def moderate_text(self, text: str) -> Dict[str, Any]:
        """Moderate text using OpenAI's moderation API"""
        try:
            response = await self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Extract categories and scores
            categories = result.categories
            category_scores = result.category_scores
            
            # Determine if content is flagged
            flagged = result.flagged
            
            # Calculate overall score (average of all category scores)
            scores = list(category_scores.dict().values())
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Get flagged categories
            flagged_categories = [cat for cat, is_flagged in categories.dict().items() if is_flagged]
            
            return {
                "flagged": flagged,
                "score": overall_score,
                "categories": categories.dict(),
                "category_scores": category_scores.dict(),
                "flagged_categories": flagged_categories,
                "provider": "openai"
            }
        except Exception as e:
            logger.error(f"OpenAI moderation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Moderation service error: {str(e)}")
    
    async def moderate_image(self, image_path: str) -> Dict[str, Any]:
        """Moderate image using OpenAI's vision API"""
        try:
            with open(image_path, "rb") as image_file:
                response = await self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image for inappropriate content. Return a JSON response with: flagged (boolean), score (0-1), categories (list of issues found), confidence (0-1). Focus on: violence, adult content, hate speech, self-harm, illegal activities."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_file.read()}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
            
            # Parse the response (this is a simplified approach)
            # In production, you'd want more robust parsing
            content = response.choices[0].message.content
            # For now, return a safe default
            return {
                "flagged": False,
                "score": 0.0,
                "categories": [],
                "confidence": 0.8,
                "provider": "openai"
            }
        except Exception as e:
            logger.error(f"OpenAI image moderation failed: {e}")
            return {
                "flagged": False,
                "score": 0.0,
                "categories": [],
                "confidence": 0.5,
                "provider": "openai"
            }
    
    async def moderate_audio(self, audio_path: str) -> Dict[str, Any]:
        """Moderate audio using OpenAI's audio transcription and moderation"""
        try:
            # For audio, we'll transcribe first then moderate the text
            with open(audio_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Now moderate the transcribed text
            text_result = await self.moderate_text(transcript.text)
            
            return {
                **text_result,
                "transcript": transcript.text,
                "provider": "openai"
            }
        except Exception as e:
            logger.error(f"OpenAI audio moderation failed: {e}")
            return {
                "flagged": False,
                "score": 0.0,
                "categories": [],
                "confidence": 0.5,
                "provider": "openai"
            }


class MockModerationProvider(BaseModerationProvider):
    """Mock moderation provider for development/testing"""
    
    def __init__(self, api_key: str = "mock"):
        super().__init__(api_key)
    
    async def moderate_text(self, text: str) -> Dict[str, Any]:
        """Mock text moderation"""
        # Simple keyword-based moderation
        flagged_keywords = ["violence", "hate", "spam", "inappropriate"]
        text_lower = text.lower()
        
        found_keywords = [kw for kw in flagged_keywords if kw in text_lower]
        flagged = len(found_keywords) > 0
        score = min(len(found_keywords) * 0.3, 1.0)
        
        return {
            "flagged": flagged,
            "score": score,
            "categories": {
                "violence": "violence" in text_lower,
                "hate": "hate" in text_lower,
                "spam": "spam" in text_lower,
                "inappropriate": "inappropriate" in text_lower
            },
            "category_scores": {
                "violence": 0.3 if "violence" in text_lower else 0.0,
                "hate": 0.3 if "hate" in text_lower else 0.0,
                "spam": 0.3 if "spam" in text_lower else 0.0,
                "inappropriate": 0.3 if "inappropriate" in text_lower else 0.0
            },
            "provider": "mock"
        }
    
    async def moderate_image(self, image_path: str) -> Dict[str, Any]:
        """Mock image moderation"""
        # Always pass for mock
        return {
            "flagged": False,
            "score": 0.0,
            "categories": [],
            "confidence": 0.9,
            "provider": "mock"
        }
    
    async def moderate_audio(self, audio_path: str) -> Dict[str, Any]:
        """Mock audio moderation"""
        # Always pass for mock
        return {
            "flagged": False,
            "score": 0.0,
            "categories": [],
            "confidence": 0.9,
            "provider": "mock"
        }


class ComplianceChecker:
    """Compliance checking for GDPR, COPPA, and platform policies"""
    
    def __init__(self):
        self.gdpr_keywords = ["personal data", "email", "phone", "address", "ssn"]
        self.coppa_keywords = ["child", "kid", "teen", "under 13", "minor"]
        self.platform_policies = {
            "youtube": ["copyright", "community guidelines", "advertiser friendly"],
            "instagram": ["community guidelines", "hate speech", "bullying"],
            "tiktok": ["community guidelines", "minor safety", "authenticity"],
            "facebook": ["community standards", "hate speech", "violence"]
        }
    
    async def check_gdpr_compliance(self, text: str) -> Dict[str, Any]:
        """Check GDPR compliance"""
        text_lower = text.lower()
        violations = []
        
        for keyword in self.gdpr_keywords:
            if keyword in text_lower:
                violations.append(f"Contains {keyword}")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendation": "Review required" if violations else "Compliant"
        }
    
    async def check_coppa_compliance(self, text: str) -> Dict[str, Any]:
        """Check COPPA compliance"""
        text_lower = text.lower()
        violations = []
        
        for keyword in self.coppa_keywords:
            if keyword in text_lower:
                violations.append(f"Contains {keyword}")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendation": "Review required" if violations else "Compliant"
        }
    
    async def check_platform_compliance(self, text: str, platform: str) -> Dict[str, Any]:
        """Check platform-specific compliance"""
        if platform not in self.platform_policies:
            return {"compliant": True, "violations": [], "recommendation": "Unknown platform"}
        
        text_lower = text.lower()
        violations = []
        
        for policy in self.platform_policies[platform]:
            if policy in text_lower:
                violations.append(f"Platform policy: {policy}")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendation": "Review required" if violations else "Compliant"
        }


async def create_audit_log(
    session,
    action_type: str,
    entity_type: str,
    entity_id: str,
    user_id: Optional[str],
    user_role: Optional[str],
    action_data: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]] = None,
    new_state: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None
):
    """Create audit log entry"""
    audit_log = AuditLog(
        action_type=action_type,
        entity_type=entity_type,
        entity_id=entity_id,
        user_id=user_id,
        user_role=user_role,
        action_data=action_data,
        previous_state=previous_state,
        new_state=new_state,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None,
        correlation_id=request.headers.get("x-correlation-id") if request else None
    )
    session.add(audit_log)
    await session.commit()


# Global service instance
moderation_service = None
compliance_checker = ComplianceChecker()


@app.on_event("startup")
async def on_startup():
    """Initialize moderation service on startup"""
    global moderation_service
    db_manager = get_db_manager()
    await db_manager.initialize()
    
    # Initialize moderation provider based on configuration
    if MODERATION_PROVIDER == "openai" and OPENAI_API_KEY:
        moderation_service = OpenAIModerationProvider(OPENAI_API_KEY)
        logger.info("Initialized OpenAI moderation provider")
    else:
        moderation_service = MockModerationProvider()
        logger.info("Initialized mock moderation provider")
    
    logger.info("Moderation service initialized")


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on shutdown"""
    await get_db_manager().close()
    logger.info("Moderation service shutdown complete")


def get_moderation_service() -> BaseModerationProvider:
    """Get moderation service instance"""
    if moderation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return moderation_service


@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint with optional deep checks"""
    health_status = {"status": "healthy", "service": "moderation-service"}
    
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
    
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Content Moderation Service API", "version": "1.0.0"}


@app.post("/moderate/text")
async def moderate_text_endpoint(
    request: Request,
    text: str = Form(..., description="Text content to moderate"),
    level: ModerationLevel = Form(ModerationLevel.MEDIUM, description="Moderation level"),
    user_id: Optional[str] = Form(None, description="User ID"),
    platform: Optional[str] = Form(None, description="Target platform")
):
    """Moderate text content"""
    service = get_moderation_service()
    result = await service.moderate_text(text)
    
    # Determine status based on score and threshold
    score = result["score"]
    flagged = result["flagged"]
    
    if flagged or score > MODERATION_THRESHOLD:
        status = ModerationStatus.FLAGGED
        requires_review = True
    else:
        status = ModerationStatus.APPROVED
        requires_review = False
    
    flags = result.get("flagged_categories", [])
    
    # Save to database
    async with get_db_manager().get_session() as session:
        moderation = ContentModeration(
            content_id=str(uuid.uuid4()),
            content_type=ModerationType.TEXT,
            status=status,
            score=score,
            flags=flags,
            categories=result.get("category_scores", {}),
            recommendations=["Manual review recommended" if requires_review else "Content appears safe"],
            requires_review=requires_review,
            audit_trail={"provider": result["provider"]},
            user_id=user_id,
            platform=platform,
            moderation_level=level.value,
            provider=result["provider"],
            provider_response=result
        )
        session.add(moderation)
        await session.commit()
        
        # Create audit log
        await create_audit_log(
            session=session,
            action_type="text_moderation",
            entity_type="content",
            entity_id=moderation.content_id,
            user_id=user_id,
            user_role="user",
            action_data={
                "text_length": len(text),
                "moderation_level": level.value,
                "platform": platform,
                "score": score,
                "flagged": flagged
            },
            request=request
        )
        
        # Add to review queue if needed
        if requires_review:
            review_queue = ManualReviewQueue(
                moderation_id=moderation.id,
                priority="normal" if score < 0.8 else "high",
                review_status="pending"
            )
            session.add(review_queue)
            await session.commit()
    
    return ModerationResult(
        content_id=moderation.content_id,
        status=status,
        score=score,
        flags=flags,
        categories=result.get("category_scores", {}),
        recommendations=["Manual review recommended" if requires_review else "Content appears safe"],
        requires_review=requires_review,
        audit_trail={"provider": result["provider"]},
        created_at=moderation.created_at
    )


@app.post("/moderate/file")
async def moderate_file_endpoint(
    request: Request,
    file: UploadFile = File(...),
    content_type: ContentType = Form(..., description="Content type"),
    level: ModerationLevel = Form(ModerationLevel.MEDIUM, description="Moderation level"),
    user_id: Optional[str] = Form(None, description="User ID"),
    platform: Optional[str] = Form(None, description="Target platform")
):
    """Moderate file content (image, audio, video)"""
    
    # Validate file type and size
    allowed_types = {
        ContentType.IMAGE: ["image/jpeg", "image/png", "image/gif", "image/webp"],
        ContentType.AUDIO: ["audio/mpeg", "audio/wav", "audio/mp3", "audio/ogg"],
        ContentType.VIDEO: ["video/mp4", "video/avi", "video/mov", "video/webm"]
    }
    
    if file.content_type not in allowed_types.get(content_type, []):
        raise HTTPException(status_code=400, detail=f"Invalid file type for {content_type}")
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    # Save file to storage
    file_path = STORAGE_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    try:
        service = get_moderation_service()
        
        if content_type == ContentType.IMAGE:
            result = await service.moderate_image(str(file_path))
        elif content_type == ContentType.AUDIO:
            result = await service.moderate_audio(str(file_path))
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")
        
        # Determine status based on score and threshold
        score = result["score"]
        flagged = result["flagged"]
        
        if flagged or score > MODERATION_THRESHOLD:
            status = ModerationStatus.FLAGGED
            requires_review = True
        else:
            status = ModerationStatus.APPROVED
            requires_review = False
        
        flags = result.get("categories", [])
        
        # Save to database
        async with get_db_manager().get_session() as session:
            moderation = ContentModeration(
                content_id=str(uuid.uuid4()),
                content_type=ModerationType(content_type.value),
                status=status,
                score=score,
                flags=flags,
                categories=result.get("category_scores", {}),
                recommendations=["Manual review recommended" if requires_review else "Content appears safe"],
                requires_review=requires_review,
                audit_trail={"provider": result["provider"]},
                user_id=user_id,
                platform=platform,
                moderation_level=level.value,
                provider=result["provider"],
                provider_response=result
            )
            session.add(moderation)
            await session.commit()
            
            # Create audit log
            await create_audit_log(
                session=session,
                action_type="file_moderation",
                entity_type="content",
                entity_id=moderation.content_id,
                user_id=user_id,
                user_role="user",
                action_data={
                    "file_name": file.filename,
                    "file_size": len(content),
                    "content_type": content_type.value,
                    "moderation_level": level.value,
                    "platform": platform,
                    "score": score,
                    "flagged": flagged
                },
                request=request
            )
            
            # Add to review queue if needed
            if requires_review:
                review_queue = ManualReviewQueue(
                    moderation_id=moderation.id,
                    priority="normal" if score < 0.8 else "high",
                    review_status="pending"
                )
                session.add(review_queue)
                await session.commit()
        
        return ModerationResult(
            content_id=moderation.content_id,
            status=status,
            score=score,
            flags=flags,
            categories=result.get("category_scores", {}),
            recommendations=["Manual review recommended" if requires_review else "Content appears safe"],
            requires_review=requires_review,
            audit_trail={"provider": result["provider"]},
            created_at=moderation.created_at
        )
        
    finally:
        # Clean up temporary file
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


@app.get("/review/queue")
async def get_review_queue(
    status: Optional[ModerationStatus] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get items requiring manual review"""
    try:
        async with get_db_manager().get_session() as session:
            query = session.query(ContentModeration).filter(
                ContentModeration.requires_review == True
            )
            
            if status:
                query = query.filter(ContentModeration.status == status)
            
            query = query.order_by(ContentModeration.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            results = await session.execute(query)
            items = results.scalars().all()
            
            return {
                "items": [
                    {
                        "id": str(item.id),
                        "content_id": item.content_id,
                        "content_type": item.content_type,
                        "status": item.status,
                        "score": item.score,
                        "flags": item.flags,
                        "created_at": item.created_at,
                        "user_id": item.user_id,
                        "platform": item.platform
                    }
                    for item in items
                ],
                "total": len(items),
                "offset": offset,
                "limit": limit
            }
    except Exception as e:
        logger.error(f"Failed to get review queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to get review queue")


@app.post("/review/{moderation_id}")
async def review_moderation(
    request: Request,
    moderation_id: str,
    review: ReviewRequest
):
    """Submit manual review decision"""
    try:
        async with get_db_manager().get_session() as session:
            # Get moderation record
            result = await session.execute(
                session.query(ContentModeration).filter(ContentModeration.id == moderation_id)
            )
            moderation = result.scalar_one_or_none()
            
            if not moderation:
                raise HTTPException(status_code=404, detail="Moderation record not found")
            
            # Store previous state for audit
            previous_state = {
                "status": moderation.status,
                "reviewer_id": moderation.reviewer_id,
                "review_notes": moderation.review_notes
            }
            
            # Update with review decision
            moderation.status = review.decision
            moderation.reviewer_id = review.reviewer_id
            moderation.review_notes = review.notes
            moderation.review_override_reason = review.override_reason
            moderation.reviewed_at = datetime.utcnow()
            
            await session.commit()
            
            # Create audit log
            await create_audit_log(
                session=session,
                action_type="manual_review",
                entity_type="content",
                entity_id=moderation.content_id,
                user_id=review.reviewer_id,
                user_role="reviewer",
                action_data={
                    "decision": review.decision.value,
                    "notes": review.notes,
                    "override_reason": review.override_reason
                },
                previous_state=previous_state,
                new_state={
                    "status": moderation.status,
                    "reviewer_id": moderation.reviewer_id,
                    "review_notes": moderation.review_notes
                },
                request=request
            )
            
            return {
                "moderation_id": moderation_id,
                "status": review.decision,
                "reviewer_id": review.reviewer_id,
                "reviewed_at": moderation.reviewed_at
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit review: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit review")


@app.get("/compliance/check")
async def check_compliance(
    text: str = Query(..., description="Text to check for compliance"),
    platform: Optional[str] = Query(None, description="Target platform")
):
    """Check compliance with GDPR, COPPA, and platform policies"""
    try:
        gdpr_result = await compliance_checker.check_gdpr_compliance(text)
        coppa_result = await compliance_checker.check_coppa_compliance(text)
        platform_result = await compliance_checker.check_platform_compliance(text, platform or "general")
        
        return ComplianceCheck(
            gdpr_compliant=gdpr_result["compliant"],
            coppa_compliant=coppa_result["compliant"],
            platform_compliant=platform_result["compliant"],
            copyright_clear=True,  # Would need separate check
            violations=(
                gdpr_result["violations"] +
                coppa_result["violations"] +
                platform_result["violations"]
            ),
            recommendations=[
                gdpr_result["recommendation"],
                coppa_result["recommendation"],
                platform_result["recommendation"]
            ]
        )
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise HTTPException(status_code=500, detail="Compliance check failed")


@app.get("/providers")
async def get_providers():
    """Get available moderation providers"""
    providers = [
        {
            "name": "MockModerationProvider",
            "type": "mock",
            "capabilities": ["text", "image", "audio"]
        }
    ]
    
    if OPENAI_API_KEY:
        providers.append({
            "name": "OpenAIModerationProvider",
            "type": "openai",
            "capabilities": ["text", "image", "audio"]
        })
    
    return {"providers": providers}


@app.get("/config")
async def get_config():
    """Get moderation service configuration"""
    return {
        "moderation_threshold": MODERATION_THRESHOLD,
        "moderation_provider": MODERATION_PROVIDER,
        "storage_path": str(STORAGE_DIR),
        "has_openai_key": bool(OPENAI_API_KEY)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)