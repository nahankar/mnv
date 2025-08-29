"""
Video Assembly Service

FastAPI service for video creation using FFmpeg and MoviePy, combining story narration,
images, and background music with multi-format support and platform-specific optimizations.
"""

import asyncio
import logging
import os
import uuid
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
import aiofiles
import numpy as np

# Video processing imports
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips, ColorClip, CompositeAudioClip, concatenate_audioclips
from moviepy.video.fx import resize, crop, colorx
import ffmpeg

from shared.database import DatabaseManager, get_db_manager
from shared.logging import get_logger
from shared.config import get_config
from shared.middleware import CorrelationMiddleware
from shared.models import Video, ContentStatus
from shared.retry import retry_api_call, convert_http_error, NetworkError
from shared.rate_limiter import get_rate_limiter

# Prometheus metrics
REQUEST_COUNT = Counter('video_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('video_service_request_duration_seconds', 'Request duration')
VIDEO_PROCESSING_TIME = Histogram('video_service_processing_duration_seconds', 'Video processing time')

app = FastAPI(title="Video Assembly Service", version="1.0.0")
logger = get_logger(__name__)
config = get_config()

# Add middleware
app.add_middleware(CorrelationMiddleware)

# Storage for temporary files
TEMP_DIR = Path("/tmp/video-assembly")
TEMP_DIR.mkdir(exist_ok=True)

# Video output directory
OUTPUT_DIR = Path("/app/videos")
OUTPUT_DIR.mkdir(exist_ok=True)


class VideoFormat(str, Enum):
    LANDSCAPE_16_9 = "16:9"      # YouTube, Facebook
    PORTRAIT_9_16 = "9:16"       # TikTok, Instagram Reels
    SQUARE_1_1 = "1:1"           # Instagram posts
    STORY_9_16 = "9:16_story"    # Instagram Stories


class VideoQuality(str, Enum):
    LOW = "low"          # 480p
    MEDIUM = "medium"    # 720p
    HIGH = "high"        # 1080p
    ULTRA = "ultra"      # 4K


class VideoEffect(str, Enum):
    NONE = "none"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    ZOOM = "zoom"
    PAN = "pan"
    COLOR_CORRECTION = "color_correction"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"


class Platform(str, Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    TWITTER = "twitter"


class VideoAsset(BaseModel):
    asset_id: str
    asset_type: str  # "image", "audio", "music"
    file_path: str
    duration: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    volume: Optional[float] = 1.0
    effects: List[VideoEffect] = Field(default_factory=list)


class VideoAssemblyRequest(BaseModel):
    story_id: str
    title: str
    description: Optional[str] = None
    format: VideoFormat = VideoFormat.LANDSCAPE_16_9
    quality: VideoQuality = VideoQuality.HIGH
    duration_limit: Optional[int] = None  # Maximum duration in seconds
    assets: List[VideoAsset] = Field(default_factory=list)
    background_music: Optional[str] = None  # Music file path
    background_volume: float = Field(0.3, ge=0.0, le=1.0)
    effects: List[VideoEffect] = Field(default_factory=list)
    platform: Platform = Platform.YOUTUBE
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VideoAssemblyResult(BaseModel):
    video_id: str
    status: ContentStatus
    file_path: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    format: VideoFormat
    quality: VideoQuality
    platform: Platform
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None
    created_at: datetime


class PlatformMetadata(BaseModel):
    title: str
    description: str
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    privacy: str = "public"
    thumbnail_path: Optional[str] = None


class VideoProcessor:
    """Main video processing orchestrator"""
    
    def __init__(self):
        self.supported_formats = {
            VideoFormat.LANDSCAPE_16_9: {"width": 1920, "height": 1080},
            VideoFormat.PORTRAIT_9_16: {"width": 1080, "height": 1920},
            VideoFormat.SQUARE_1_1: {"width": 1080, "height": 1080},
            VideoFormat.STORY_9_16: {"width": 1080, "height": 1920}
        }
        
        self.quality_settings = {
            VideoQuality.LOW: {"bitrate": "1000k", "crf": 28},
            VideoQuality.MEDIUM: {"bitrate": "2000k", "crf": 23},
            VideoQuality.HIGH: {"bitrate": "4000k", "crf": 18},
            VideoQuality.ULTRA: {"bitrate": "8000k", "crf": 15}
        }
    
    async def assemble_video(self, request: VideoAssemblyRequest) -> VideoAssemblyResult:
        """Main video assembly workflow"""
        start_time = datetime.utcnow()
        video_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting video assembly for {video_id}")
            
            # Create temporary working directory
            work_dir = TEMP_DIR / video_id
            work_dir.mkdir(exist_ok=True)
            
            # Process assets
            processed_assets = await self._process_assets(request.assets, work_dir)
            
            # Create video composition
            video_clip = await self._create_video_composition(
                processed_assets, request, work_dir
            )
            
            # Apply effects
            if request.effects:
                video_clip = await self._apply_effects(video_clip, request.effects)
            
            # Apply duration limit if specified
            if request.duration_limit:
                video_clip = video_clip.subclip(0, min(video_clip.duration, request.duration_limit))
            
            # Generate platform-specific metadata
            platform_metadata = await self._generate_platform_metadata(request)
            
            # Export video
            output_path = await self._export_video(
                video_clip, request, work_dir, video_id
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Get file size
            file_size = output_path.stat().st_size if output_path.exists() else 0
            
            result = VideoAssemblyResult(
                video_id=video_id,
                status=ContentStatus.COMPLETED,
                file_path=str(output_path),
                duration=video_clip.duration,
                file_size=file_size,
                format=request.format,
                quality=request.quality,
                platform=request.platform,
                metadata=platform_metadata.dict(),
                processing_time=processing_time,
                created_at=datetime.utcnow()
            )
            
            # Clean up temporary files
            await self._cleanup_temp_files(work_dir)
            
            logger.info(f"Video assembly completed for {video_id}")
            return result
            
        except Exception as e:
            logger.error(f"Video assembly failed for {video_id}: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return VideoAssemblyResult(
                video_id=video_id,
                status=ContentStatus.FAILED,
                format=request.format,
                quality=request.quality,
                platform=request.platform,
                processing_time=processing_time,
                created_at=datetime.utcnow()
            )
    
    async def _process_assets(self, assets: List[VideoAsset], work_dir: Path) -> List[Dict[str, Any]]:
        """Process and prepare video assets"""
        processed_assets = []
        
        for asset in assets:
            try:
                if asset.asset_type == "image":
                    processed = await self._process_image_asset(asset, work_dir)
                elif asset.asset_type == "audio":
                    processed = await self._process_audio_asset(asset, work_dir)
                elif asset.asset_type == "music":
                    processed = await self._process_music_asset(asset, work_dir)
                else:
                    logger.warning(f"Unknown asset type: {asset.asset_type}")
                    continue
                
                processed_assets.append(processed)
                
            except Exception as e:
                logger.error(f"Failed to process asset {asset.asset_id}: {e}")
                continue
        
        return processed_assets
    
    async def _process_image_asset(self, asset: VideoAsset, work_dir: Path) -> Dict[str, Any]:
        """Process image asset for video composition"""
        # Load image and create clip
        image_clip = ImageClip(asset.file_path)
        
        # Set duration if not specified
        if asset.duration is None:
            asset.duration = 3.0  # Default 3 seconds per image
        
        image_clip = image_clip.set_duration(asset.duration)
        
        # Apply timing if specified
        if asset.start_time is not None:
            image_clip = image_clip.set_start(asset.start_time)
        
        return {
            "type": "image",
            "clip": image_clip,
            "asset": asset
        }
    
    async def _process_audio_asset(self, asset: VideoAsset, work_dir: Path) -> Dict[str, Any]:
        """Process audio asset for video composition"""
        # Load audio clip
        audio_clip = AudioFileClip(asset.file_path)
        
        # Apply volume adjustment
        if asset.volume != 1.0:
            audio_clip = audio_clip.volumex(asset.volume)
        
        # Apply timing if specified
        if asset.start_time is not None:
            audio_clip = audio_clip.set_start(asset.start_time)
        
        # Apply duration limits if specified
        if asset.end_time is not None:
            audio_clip = audio_clip.subclip(0, asset.end_time)
        
        return {
            "type": "audio",
            "clip": audio_clip,
            "asset": asset
        }
    
    async def _process_music_asset(self, asset: VideoAsset, work_dir: Path) -> Dict[str, Any]:
        """Process background music asset"""
        # Load music clip
        music_clip = AudioFileClip(asset.file_path)
        
        # Apply volume adjustment
        if asset.volume != 1.0:
            music_clip = music_clip.volumex(asset.volume)
        
        return {
            "type": "music",
            "clip": music_clip,
            "asset": asset
        }
    
    async def _create_video_composition(
        self, 
        processed_assets: List[Dict[str, Any]], 
        request: VideoAssemblyRequest,
        work_dir: Path
    ) -> CompositeVideoClip:
        """Create the main video composition"""
        
        # Separate assets by type
        image_assets = [a for a in processed_assets if a["type"] == "image"]
        audio_assets = [a for a in processed_assets if a["type"] == "audio"]
        music_assets = [a for a in processed_assets if a["type"] == "music"]
        
        # Create video clips from images
        video_clips = []
        for asset in image_assets:
            clip = asset["clip"]
            
            # Resize to target format
            target_size = self.supported_formats[request.format]
            clip = clip.resize((target_size["width"], target_size["height"]))
            
            video_clips.append(clip)
        
        # Concatenate video clips
        if video_clips:
            main_video = concatenate_videoclips(video_clips, method="compose")
        else:
            # Create a black video if no images
            target_size = self.supported_formats[request.format]
            main_video = ColorClip(
                size=(target_size["width"], target_size["height"]),
                color=(0, 0, 0),
                duration=10.0  # Default 10 seconds
            )
        
        # Add audio tracks
        audio_clips = []
        
        # Add narration audio
        for asset in audio_assets:
            audio_clips.append(asset["clip"])
        
        # Add background music
        if music_assets:
            music_clip = music_assets[0]["clip"]
            # Loop music to match video duration
            if music_clip.duration < main_video.duration:
                loops_needed = int(main_video.duration / music_clip.duration) + 1
                music_clip = concatenate_audioclips([music_clip] * loops_needed)
            
            # Trim to video duration
            music_clip = music_clip.subclip(0, main_video.duration)
            audio_clips.append(music_clip)
        
        # Combine audio tracks
        if audio_clips:
            final_audio = CompositeAudioClip(audio_clips)
            main_video = main_video.set_audio(final_audio)
        
        return main_video
    
    async def _apply_effects(self, video_clip: VideoFileClip, effects: List[VideoEffect]) -> VideoFileClip:
        """Apply video effects"""
        for effect in effects:
            try:
                if effect == VideoEffect.FADE_IN:
                    video_clip = video_clip.fadein(1.0)
                elif effect == VideoEffect.FADE_OUT:
                    video_clip = video_clip.fadeout(1.0)
                elif effect == VideoEffect.ZOOM:
                    video_clip = video_clip.fx(resize, 1.2)
                elif effect == VideoEffect.COLOR_CORRECTION:
                    video_clip = video_clip.fx(colorx, 1.1)
                elif effect == VideoEffect.BRIGHTNESS:
                    video_clip = video_clip.fx(colorx, 1.2)
                elif effect == VideoEffect.CONTRAST:
                    video_clip = video_clip.fx(colorx, 1.3)
                
            except Exception as e:
                logger.warning(f"Failed to apply effect {effect}: {e}")
                continue
        
        return video_clip
    
    @retry_api_call(max_retries=3)
    async def _export_video(
        self, 
        video_clip: VideoFileClip, 
        request: VideoAssemblyRequest,
        work_dir: Path,
        video_id: str
    ) -> Path:
        """Export video to final format"""
        
        # Determine output format based on platform
        output_format = self._get_output_format(request.platform)
        quality_settings = self.quality_settings[request.quality]
        
        # Generate output filename
        output_filename = f"{video_id}_{request.format}_{request.quality}.{output_format}"
        output_path = OUTPUT_DIR / output_filename
        
        # Export video
        video_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            bitrate=quality_settings["bitrate"],
            fps=30,
            threads=4,
            verbose=False,
            logger=None
        )
        
        return output_path
    
    def _get_output_format(self, platform: Platform) -> str:
        """Get optimal output format for platform"""
        if platform == Platform.YOUTUBE:
            return "mp4"
        elif platform == Platform.INSTAGRAM:
            return "mp4"
        elif platform == Platform.TIKTOK:
            return "mp4"
        elif platform == Platform.FACEBOOK:
            return "mp4"
        elif platform == Platform.TWITTER:
            return "mp4"
        else:
            return "mp4"
    
    async def _generate_platform_metadata(self, request: VideoAssemblyRequest) -> PlatformMetadata:
        """Generate platform-specific metadata"""
        
        # Base metadata
        metadata = PlatformMetadata(
            title=request.title,
            description=request.description or "",
            tags=[],
            privacy="public"
        )
        
        # Platform-specific customizations
        if request.platform == Platform.YOUTUBE:
            metadata.tags = self._generate_youtube_tags(request.title, request.metadata)
            metadata.category = "Entertainment"
            
        elif request.platform == Platform.INSTAGRAM:
            metadata.tags = self._generate_instagram_tags(request.title, request.metadata)
            
        elif request.platform == Platform.TIKTOK:
            metadata.tags = self._generate_tiktok_tags(request.title, request.metadata)
            
        elif request.platform == Platform.FACEBOOK:
            metadata.tags = self._generate_facebook_tags(request.title, request.metadata)
        
        return metadata
    
    def _generate_youtube_tags(self, title: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate YouTube-optimized tags"""
        base_tags = ["story", "ai generated", "creative"]
        
        # Add genre-specific tags
        if "genre" in metadata:
            base_tags.append(metadata["genre"])
        
        # Add mood-specific tags
        if "mood" in metadata:
            base_tags.append(metadata["mood"])
        
        # Add trending tags
        trending_tags = ["#shorts", "#storytime", "#creative"]
        base_tags.extend(trending_tags)
        
        return base_tags[:15]  # YouTube limit
    
    def _generate_instagram_tags(self, title: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate Instagram-optimized tags"""
        base_tags = ["story", "creative", "ai"]
        
        # Add relevant hashtags
        hashtags = ["#story", "#creative", "#ai", "#digitalart"]
        base_tags.extend(hashtags)
        
        return base_tags[:30]  # Instagram limit
    
    def _generate_tiktok_tags(self, title: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate TikTok-optimized tags"""
        base_tags = ["story", "fyp", "viral"]
        
        # Add trending tags
        trending_tags = ["#fyp", "#foryou", "#story", "#viral"]
        base_tags.extend(trending_tags)
        
        return base_tags[:20]  # TikTok limit
    
    def _generate_facebook_tags(self, title: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate Facebook-optimized tags"""
        base_tags = ["story", "creative", "entertainment"]
        
        # Add engagement tags
        engagement_tags = ["#story", "#creative", "#entertainment"]
        base_tags.extend(engagement_tags)
        
        return base_tags[:10]  # Facebook limit
    
    async def _cleanup_temp_files(self, work_dir: Path):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(work_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


class VideoService:
    """Main video service orchestrator"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.processor = VideoProcessor()
    
    async def create_video(self, request: VideoAssemblyRequest) -> VideoAssemblyResult:
        """Create video with full processing pipeline"""
        
        # Validate request
        await self._validate_request(request)
        
        # Process video assembly
        result = await self.processor.assemble_video(request)
        
        # Store video record
        await self._store_video_record(request, result)
        
        return result
    
    async def _validate_request(self, request: VideoAssemblyRequest):
        """Validate video assembly request"""
        
        # Check if assets are provided
        if not request.assets:
            raise HTTPException(status_code=400, detail="At least one asset is required")
        
        # Validate asset files exist
        for asset in request.assets:
            if not Path(asset.file_path).exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Asset file not found: {asset.file_path}"
                )
        
        # Validate duration limit
        if request.duration_limit and request.duration_limit > 600:  # 10 minutes
            raise HTTPException(
                status_code=400, 
                detail="Duration limit cannot exceed 600 seconds"
            )
    
    async def _store_video_record(self, request: VideoAssemblyRequest, result: VideoAssemblyResult):
        """Store video record in database"""
        try:
            async with self.db_manager.get_session() as session:
                # Derive resolution from selected format
                size = self.processor.supported_formats[request.format]
                resolution = f"{size['width']}x{size['height']}"
                
                video = Video(
                    id=uuid.uuid4(),
                    story_id=request.story_id,
                    title=request.title,
                    description=request.description,
                    file_path=result.file_path,
                    duration=result.duration,
                    file_size=result.file_size,
                    format_type=request.format.value,
                    resolution=resolution,
                    target_platforms=[request.platform.value],
                    status=result.status,
                    assembly_parameters=result.metadata,
                    processing_time=result.processing_time
                )
                session.add(video)
                await session.commit()
                await session.refresh(video)
        except Exception as e:
            logger.error(f"Failed to store video record: {e}")


# Global service instance
video_service: Optional[VideoService] = None


@app.on_event("startup")
async def on_startup():
    """Initialize video service on startup"""
    global video_service
    db_manager = get_db_manager()
    await db_manager.initialize()
    video_service = VideoService(db_manager)
    logger.info("Video service initialized")


@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on shutdown"""
    await get_db_manager().close()
    logger.info("Video service shutdown complete")


def get_video_service() -> VideoService:
    """Get video service instance"""
    if video_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return video_service


@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint with optional deep checks"""
    health_status = {"status": "healthy", "service": "video-service"}
    
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
        
        # Check FFmpeg availability
        try:
            import subprocess
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                health_status["ffmpeg"] = "available"
            else:
                health_status["ffmpeg"] = "error"
        except Exception as e:
            health_status["ffmpeg"] = f"error: {str(e)}"
    
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Video Assembly Service API", "version": "1.0.0"}


@app.post("/assemble", response_model=VideoAssemblyResult)
async def assemble_video_endpoint(
    request: VideoAssemblyRequest,
    service: VideoService = Depends(get_video_service)
):
    """Assemble video from assets"""
    try:
        with REQUEST_DURATION.time():
            result = await service.create_video(request)
        
        REQUEST_COUNT.labels(method="POST", endpoint="/assemble", status="success").inc()
        return result
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/assemble", status="error").inc()
        logger.error(f"Video assembly failed: {e}")
        raise HTTPException(status_code=500, detail="Video assembly failed")


@app.post("/assemble/async")
async def assemble_video_async_endpoint(
    request: VideoAssemblyRequest,
    background_tasks: BackgroundTasks,
    service: VideoService = Depends(get_video_service)
):
    """Assemble video asynchronously"""
    video_id = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(service.create_video, request)
    
    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Video assembly started in background"
    }


@app.get("/videos/{video_id}")
async def get_video_status(
    video_id: str,
    service: VideoService = Depends(get_video_service)
):
    """Get video status and details"""
    try:
        async with get_db_manager().get_session() as session:
            result = await session.execute(
                session.query(Video).filter(Video.id == video_id)
            )
            video = result.scalar_one_or_none()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            return {
                "video_id": str(video.id),
                "status": video.status,
                "title": video.title,
                "duration": video.duration,
                "file_size": video.file_size,
                "format": video.format_type,
                "resolution": video.resolution,
                "platforms": video.target_platforms,
                "processing_time": video.processing_time,
                "created_at": video.created_at
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get video status")


@app.get("/formats")
async def get_supported_formats():
    """Get supported video formats and quality settings"""
    processor = VideoProcessor()
    
    return {
        "formats": {
            format_name: {
                "width": settings["width"],
                "height": settings["height"],
                "aspect_ratio": f"{settings['width']}:{settings['height']}"
            }
            for format_name, settings in processor.supported_formats.items()
        },
        "quality_settings": processor.quality_settings,
        "platforms": [platform.value for platform in Platform],
        "effects": [effect.value for effect in VideoEffect]
    }


@app.get("/platforms/{platform}/metadata")
async def get_platform_metadata_template(platform: Platform):
    """Get metadata template for specific platform"""
    
    templates = {
        Platform.YOUTUBE: {
            "title_max_length": 100,
            "description_max_length": 5000,
            "tags_max_count": 15,
            "categories": ["Entertainment", "Education", "Music", "Gaming", "News"],
            "privacy_options": ["public", "unlisted", "private"]
        },
        Platform.INSTAGRAM: {
            "title_max_length": 125,
            "description_max_length": 2200,
            "tags_max_count": 30,
            "formats": ["1:1", "9:16", "16:9"],
            "privacy_options": ["public", "private"]
        },
        Platform.TIKTOK: {
            "title_max_length": 150,
            "description_max_length": 2200,
            "tags_max_count": 20,
            "formats": ["9:16"],
            "privacy_options": ["public", "friends", "private"]
        },
        Platform.FACEBOOK: {
            "title_max_length": 100,
            "description_max_length": 63206,
            "tags_max_count": 10,
            "formats": ["16:9", "1:1", "9:16"],
            "privacy_options": ["public", "friends", "private"]
        }
    }
    
    return templates.get(platform, {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)