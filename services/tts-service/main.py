from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
import asyncio
import aiofiles
import os
import uuid
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

from shared.database import get_db_connection, get_db_manager
from shared.logging import get_logger
from shared.config import get_config
from shared.middleware import CorrelationMiddleware
from shared.retry import retry_api_call
from shared.rate_limiter import RateLimiter

# Prometheus metrics
REQUEST_COUNT = Counter('tts_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('tts_service_request_duration_seconds', 'Request duration')
PROVIDER_CALLS = Counter('tts_service_provider_calls_total', 'Provider API calls', ['provider', 'status'])
SYNTHESIS_DURATION = Histogram('tts_service_synthesis_duration_seconds', 'TTS synthesis duration')

from providers.elevenlabs_provider import ElevenLabsProvider
from providers.openai_provider import OpenAITTSProvider
from providers.azure_provider import AzureTTSProvider
from audio_processor import AudioProcessor

app = FastAPI(title="TTS Service", version="1.0.0")
logger = get_logger(__name__)
config = get_config()

# Add middleware
app.add_middleware(CorrelationMiddleware)

# Initialize providers
elevenlabs_provider = ElevenLabsProvider()
openai_provider = OpenAITTSProvider()
azure_provider = AzureTTSProvider()
audio_processor = AudioProcessor()

# Rate limiter
rate_limiter = RateLimiter()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await rate_limiter.initialize()
    logger.info("TTS service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await rate_limiter.close()
    logger.info("TTS service shutdown complete")

# Provider fallback order
PROVIDER_FALLBACK_ORDER = [
    ("elevenlabs", elevenlabs_provider),
    ("openai", openai_provider),
    ("azure", azure_provider)
]

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", max_length=5000)
    voice_id: Optional[str] = Field(None, description="Specific voice ID to use")
    provider: Optional[str] = Field(None, description="Preferred TTS provider")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
    pitch: float = Field(1.0, description="Pitch adjustment", ge=0.5, le=2.0)
    volume: float = Field(1.0, description="Volume level", ge=0.1, le=2.0)
    format: str = Field("mp3", description="Output audio format")
    quality: str = Field("standard", description="Audio quality level")

class TTSResponse(BaseModel):
    audio_id: str
    file_path: str
    duration_seconds: float
    provider_used: str
    voice_used: str
    format: str
    metadata: Dict[str, Any]

class Voice(BaseModel):
    id: str
    name: str
    provider: str
    language: str
    gender: Optional[str]
    description: Optional[str]
    preview_url: Optional[str]

class AudioProcessingRequest(BaseModel):
    audio_id: str
    operations: List[Dict[str, Any]]

@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint with optional deep checks"""
    health_status = {"status": "healthy", "service": "tts-service"}
    
    if deep:
        # Deep health check - test database and Redis connectivity
        try:
            db_manager = get_db_manager()
            async with db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
            health_status["database"] = "connected"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["database"] = f"error: {str(e)}"
        
        try:
            # Test Redis connectivity if available
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
    return {"message": "TTS Service API", "version": "1.0.0"}

@app.get("/providers")
async def get_providers():
    """Get available TTS providers and their status"""
    providers = []
    for name, provider in PROVIDER_FALLBACK_ORDER:
        try:
            status = await provider.health_check()
            providers.append({
                "name": name,
                "status": "healthy" if status else "unhealthy",
                "capabilities": provider.get_capabilities()
            })
        except Exception as e:
            logger.error(f"Provider {name} health check failed: {e}")
            providers.append({
                "name": name,
                "status": "unhealthy",
                "error": str(e)
            })
    
    return {"providers": providers}

@app.get("/voices/available")
async def get_available_voices(provider: Optional[str] = None) -> List[Voice]:
    """Retrieve available voice options from all or specific provider"""
    voices = []
    
    providers_to_check = PROVIDER_FALLBACK_ORDER
    if provider:
        providers_to_check = [(p[0], p[1]) for p in PROVIDER_FALLBACK_ORDER if p[0] == provider]
    
    for provider_name, provider_instance in providers_to_check:
        try:
            provider_voices = await provider_instance.get_voices()
            voices.extend(provider_voices)
        except Exception as e:
            logger.error(f"Failed to get voices from {provider_name}: {e}")
    
    return voices

@app.post("/synthesize/speech")
async def synthesize_speech(request: TTSRequest) -> TTSResponse:
    """Convert text to speech with voice customization and provider fallback"""
    
    # Rate limiting check
    client_id = "tts-service"  # In production, extract from auth token
    if not await rate_limiter.is_allowed(f"tts:{client_id}", limit=100, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    audio_id = str(uuid.uuid4())
    logger.info(f"Starting TTS synthesis for audio_id: {audio_id}")
    
    # Determine provider order
    providers_to_try = PROVIDER_FALLBACK_ORDER.copy()
    if request.provider:
        # Move preferred provider to front
        preferred = [(name, prov) for name, prov in providers_to_try if name == request.provider]
        others = [(name, prov) for name, prov in providers_to_try if name != request.provider]
        providers_to_try = preferred + others
    
    last_error = None
    
    for provider_name, provider_instance in providers_to_try:
        try:
            logger.info(f"Attempting TTS with provider: {provider_name}")
            
            # Generate audio with retry logic
            @retry_api_call(max_retries=3)
            async def synthesize_with_retry():
                return await provider_instance.synthesize_speech(
                    text=request.text,
                    voice_id=request.voice_id,
                    speed=request.speed,
                    pitch=request.pitch,
                    format=request.format,
                    quality=request.quality
                )
            
            audio_result = await synthesize_with_retry()
            
            # Extract audio data from result
            audio_data = audio_result.get("audio_data")
            if not audio_data:
                raise Exception("No audio data returned from provider")
            
            # Process audio if needed
            if request.volume != 1.0:
                audio_data = await audio_processor.adjust_volume(
                    audio_data, request.volume
                )
            
            # Save audio file
            file_path = await _save_audio_file(audio_id, audio_data, request.format)
            
            # Get audio duration
            duration = await audio_processor.get_duration(file_path)
            
            response = TTSResponse(
                audio_id=audio_id,
                file_path=file_path,
                duration_seconds=duration,
                provider_used=provider_name,
                voice_used=audio_result.get("voice_used", request.voice_id or "default"),
                format=request.format,
                metadata={
                    "text_length": len(request.text),
                    "speed": request.speed,
                    "pitch": request.pitch,
                    "volume": request.volume,
                    "quality": request.quality,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"TTS synthesis completed successfully with {provider_name}")
            return response
            
        except Exception as e:
            logger.error(f"TTS failed with provider {provider_name}: {e}")
            last_error = e
            continue
    
    # All providers failed
    logger.error(f"All TTS providers failed for audio_id: {audio_id}")
    raise HTTPException(
        status_code=503,
        detail=f"TTS synthesis failed with all providers. Last error: {str(last_error)}"
    )

@app.post("/process/audio")
async def process_audio(request: AudioProcessingRequest):
    """Apply audio processing operations to existing audio file"""
    
    try:
        # Load existing audio file
        file_path = f"/tmp/audio/{request.audio_id}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        processed_audio = file_path
        
        # Apply processing operations
        for operation in request.operations:
            op_type = operation.get("type")
            
            if op_type == "normalize":
                processed_audio = await audio_processor.normalize_audio(processed_audio)
            elif op_type == "trim_silence":
                processed_audio = await audio_processor.trim_silence(processed_audio)
            elif op_type == "adjust_speed":
                speed = operation.get("speed", 1.0)
                processed_audio = await audio_processor.adjust_speed(processed_audio, speed)
            elif op_type == "convert_format":
                target_format = operation.get("format", "mp3")
                processed_audio = await audio_processor.convert_format(processed_audio, target_format)
            else:
                logger.warning(f"Unknown audio processing operation: {op_type}")
        
        # Get updated duration
        duration = await audio_processor.get_duration(processed_audio)
        
        return {
            "audio_id": request.audio_id,
            "file_path": processed_audio,
            "duration_seconds": duration,
            "operations_applied": len(request.operations)
        }
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@app.get("/audio/{audio_id}")
async def get_audio_file(audio_id: str):
    """Download audio file by ID"""
    
    # Find audio file
    audio_dir = Path("/tmp/audio")
    audio_files = list(audio_dir.glob(f"{audio_id}.*"))
    
    if not audio_files:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    file_path = audio_files[0]
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=f"{audio_id}.{file_path.suffix[1:]}"
    )

@app.delete("/audio/{audio_id}")
async def delete_audio_file(audio_id: str):
    """Delete audio file by ID"""
    
    try:
        audio_dir = Path("/tmp/audio")
        audio_files = list(audio_dir.glob(f"{audio_id}.*"))
        
        deleted_count = 0
        for file_path in audio_files:
            file_path.unlink()
            deleted_count += 1
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return {"message": f"Deleted {deleted_count} audio file(s)", "audio_id": audio_id}
        
    except Exception as e:
        logger.error(f"Failed to delete audio file {audio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete audio file")

async def _save_audio_file(audio_id: str, audio_data: bytes, format: str) -> str:
    """Save audio data to file system"""
    
    # Ensure audio directory exists
    audio_dir = Path("/tmp/audio")
    audio_dir.mkdir(exist_ok=True)
    
    file_path = audio_dir / f"{audio_id}.{format}"
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(audio_data)
    
    return str(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)