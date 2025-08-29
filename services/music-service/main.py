"""
Music Generation Service

FastAPI service for generating AI background music (Suno/Mubert) with
mock provider for local/dev, licensing evidence tracking, and storage.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime
from pathlib import Path
import os
import uuid
import aiofiles
import asyncio
import subprocess
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from shared.database import get_db_manager
from shared.logging import get_logger
from shared.config import get_config
from shared.middleware import CorrelationMiddleware
from shared.models import MediaAsset, MediaType
from shared.retry import retry_api_call, convert_http_error, NetworkError
from shared.rate_limiter import get_rate_limiter

# Metrics
REQUEST_COUNT = Counter('music_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('music_service_request_duration_seconds', 'Request duration')
PROVIDER_CALLS = Counter('music_service_provider_calls_total', 'Provider API calls', ['provider', 'status'])

app = FastAPI(title="Music Service", version="1.0.0")
logger = get_logger(__name__)
config = get_config()

# Add middleware
app.add_middleware(CorrelationMiddleware)

# Storage
UPLOAD_DIR = Path(os.getenv("MUSIC_UPLOAD_DIR", "/app/uploads/music"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class MusicRequest(BaseModel):
    genre: str = Field("ambient", description="Music genre")
    mood: str = Field("calm", description="Mood of the background music")
    duration_seconds: int = Field(60, ge=5, le=600, description="Target duration in seconds")
    provider: Optional[str] = Field("mock", description="music provider: mock|mubert|suno")


class MusicResponse(BaseModel):
    id: str
    file_path: str
    file_url: str
    provider_used: str
    duration_seconds: int
    metadata: Dict
    created_at: datetime


class BaseMusicProvider:
    async def generate(self, genre: str, mood: str, duration_seconds: int) -> Dict:
        raise NotImplementedError


class AudioProcessor:
    """Audio processing utilities for normalization and duration adjustment"""
    
    async def normalize_volume(self, file_path: str, target_lufs: float = -14.0) -> str:
        """Normalize audio to target LUFS using ffmpeg"""
        try:
            output_path = file_path.replace('.wav', '_normalized.wav').replace('.mp3', '_normalized.mp3')
            cmd = [
                'ffmpeg', '-i', file_path,
                '-af', f'loudnorm=I={target_lufs}:LRA=11:TP=-1.5',
                '-y', output_path
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.warning(f"FFmpeg normalization failed: {stderr.decode()}")
                return file_path  # Return original if normalization fails
            return output_path
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return file_path
    
    async def adjust_duration(self, file_path: str, target_duration: int) -> str:
        """Adjust audio duration to target length (loop or trim)"""
        try:
            output_path = file_path.replace('.wav', '_adjusted.wav').replace('.mp3', '_adjusted.mp3')
            # Get current duration
            cmd_duration = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', file_path]
            process = await asyncio.create_subprocess_exec(*cmd_duration, stdout=asyncio.subprocess.PIPE)
            stdout, _ = await process.communicate()
            current_duration = float(stdout.decode().strip())
            
            if current_duration < target_duration:
                # Loop audio to reach target duration
                cmd = [
                    'ffmpeg', '-stream_loop', '-1', '-i', file_path,
                    '-t', str(target_duration),
                    '-y', output_path
                ]
            else:
                # Trim audio to target duration
                cmd = [
                    'ffmpeg', '-i', file_path,
                    '-t', str(target_duration),
                    '-y', output_path
                ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.warning(f"Duration adjustment failed: {stderr.decode()}")
                return file_path
            return output_path
        except Exception as e:
            logger.warning(f"Duration adjustment failed: {e}")
            return file_path
    
    async def adjust_volume(self, file_path: str, volume_db: float = -20.0) -> str:
        """Adjust volume for background music (typically lower than narration)"""
        try:
            output_path = file_path.replace('.wav', '_volume.wav').replace('.mp3', '_volume.mp3')
            cmd = [
                'ffmpeg', '-i', file_path,
                '-af', f'volume={volume_db}dB',
                '-y', output_path
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.warning(f"Volume adjustment failed: {stderr.decode()}")
                return file_path
            return output_path
        except Exception as e:
            logger.warning(f"Volume adjustment failed: {e}")
            return file_path


class MockMusicProvider(BaseMusicProvider):
    async def generate(self, genre: str, mood: str, duration_seconds: int) -> Dict:
        """Generate silent audio bytes with minimal WAV header using pydub."""
        from pydub import AudioSegment
        # 1 second of silence scaled to requested duration
        silent = AudioSegment.silent(duration=duration_seconds * 1000)
        tmp_id = str(uuid.uuid4())
        tmp_path = UPLOAD_DIR / f"{tmp_id}.wav"
        silent.export(tmp_path, format="wav")
        return {
            "provider": "mock",
            "file_path": str(tmp_path),
            "format": "wav",
            "license": {
                "type": "mock-dev",
                "terms": "development-only",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }


class MubertProvider(BaseMusicProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mubert.com/v2/Track"
    
    @retry_api_call(max_retries=3)
    async def generate(self, genre: str, mood: str, duration_seconds: int) -> Dict:
        import httpx
        params = {
            "apikey": self.api_key,
            "genre": genre,
            "mood": mood,
            "duration": duration_seconds
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise convert_http_error(e.response.status_code, str(e))
            except httpx.RequestError as e:
                raise NetworkError(f"Network error calling Mubert: {str(e)}")
            music_id = str(uuid.uuid4())
            out_path = UPLOAD_DIR / f"{music_id}.mp3"
            async with aiofiles.open(out_path, 'wb') as f:
                await f.write(resp.content)
            return {
                "provider": "mubert",
                "file_path": str(out_path),
                "format": "mp3",
                "license": {
                    "type": "mubert",
                    "receipt": resp.headers.get("X-Request-ID", "unknown"),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }


class SunoProvider(BaseMusicProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.suno.ai/generate"  # placeholder
    
    @retry_api_call(max_retries=3)
    async def generate(self, genre: str, mood: str, duration_seconds: int) -> Dict:
        # Placeholder stub with 501 to indicate incomplete provider
        raise HTTPException(status_code=501, detail="Suno provider not implemented yet")


def get_provider(name: str) -> BaseMusicProvider:
    name = (name or "mock").lower()
    if name == "mock":
        return MockMusicProvider()
    if name == "mubert":
        key = os.getenv("MUBERT_API_KEY")
        if not key:
            raise HTTPException(status_code=400, detail="Missing MUBERT_API_KEY")
        return MubertProvider(key)
    if name == "suno":
        key = os.getenv("SUNO_API_KEY")
        if not key:
            raise HTTPException(status_code=400, detail="Missing SUNO_API_KEY")
        return SunoProvider(key)
    raise HTTPException(status_code=400, detail=f"Unknown provider '{name}'")


audio_processor = AudioProcessor()


@app.get("/health")
async def health_check(deep: bool = False):
    health = {"status": "healthy", "service": "music-service"}
    if deep:
        try:
            db = get_db_manager()
            async with db.get_session() as session:
                await session.execute("SELECT 1")
            health["database"] = "connected"
        except Exception as e:
            health["status"] = "unhealthy"
            health["database"] = f"error: {e}"
    return health


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {"message": "Music Service API", "version": "1.0.0"}


@app.post("/generate/music", response_model=MusicResponse)
async def generate_music(req: MusicRequest):
    provider = get_provider(req.provider or "mock")
    # Rate limit and cost estimation
    limiter = await get_rate_limiter()
    user_id = os.getenv("SERVICE_NAME", "music-service")
    if not await limiter.check_user_limit(user_id, "music_generation"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        with REQUEST_DURATION.time():
            result = await provider.generate(req.genre, req.mood, req.duration_seconds)
        PROVIDER_CALLS.labels(provider=result["provider"], status="success").inc()

        # Process audio: normalize, adjust duration, and set background volume
        file_path = result["file_path"]
        file_path = await audio_processor.normalize_volume(file_path)
        file_path = await audio_processor.adjust_duration(file_path, req.duration_seconds)
        file_path = await audio_processor.adjust_volume(file_path, volume_db=-20.0)  # Background music level

        # If provider already wrote file, use it; otherwise write bytes (mock writes file)
        file_path = Path(file_path).resolve()
        if not str(file_path).startswith(str(UPLOAD_DIR.resolve())):
            raise HTTPException(status_code=500, detail="Invalid storage path")

        music_id = file_path.stem

        # Rough cost estimation per second (placeholder values)
        cost_per_second = {
            "mock": 0.0,
            "mubert": 0.0005,
            "suno": 0.001,
        }.get(result["provider"], 0.0005)
        est_cost = req.duration_seconds * cost_per_second
        await limiter.track_cost(user_id, "music_generation", est_cost)

        # Persist record (optional: only if DB available)
        created_at = datetime.utcnow()
        try:
            db = get_db_manager()
            async with db.get_session() as session:
                asset = MediaAsset(
                    id=uuid.UUID(music_id),
                    asset_type=MediaType.MUSIC,
                    file_path=str(file_path),
                    metadata={
                        "genre": req.genre,
                        "mood": req.mood,
                        "duration_seconds": req.duration_seconds,
                        **result,
                    },
                )
                session.add(asset)
                await session.commit()
                await session.refresh(asset)
                created_at = asset.created_at
        except Exception as e:
            logger.warning(f"Music asset DB persist skipped: {e}")

        return MusicResponse(
            id=music_id,
            file_path=str(file_path),
            file_url=f"/music/{file_path.name}",
            provider_used=result["provider"],
            duration_seconds=req.duration_seconds,
            metadata=result,
            created_at=created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        PROVIDER_CALLS.labels(provider=req.provider or "mock", status="error").inc()
        logger.error(f"Music generation failed: {e}")
        raise HTTPException(status_code=500, detail="Music generation failed")


@app.get("/providers")
async def get_providers():
    return {
        "providers": [
            {"name": "mock", "status": "healthy", "cost_hint_per_second": 0.0},
            {"name": "mubert", "status": "unknown", "cost_hint_per_second": 0.0005},
            {"name": "suno", "status": "unimplemented", "cost_hint_per_second": 0.001},
        ]
    }


@app.get("/music/{filename}")
async def get_music(filename: str):
    try:
        path = (UPLOAD_DIR / filename).resolve()
        if not path.is_file() or not str(path).startswith(str(UPLOAD_DIR.resolve())):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path)
    except Exception:
        raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)