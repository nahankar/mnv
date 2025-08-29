from __future__ import annotations

import os
import uuid
from typing import Dict, Any, List, Tuple
import hashlib
import time

import httpx
from prefect import flow, task, get_run_logger

from shared.rate_limiter import get_rate_limiter
from shared.database import get_db_manager
from shared.models import ModelConfiguration
from sqlalchemy import select
from prometheus_client import Counter, Histogram

BASES = {
    "story": os.getenv("STORY_BASE", "http://story-service:8001"),
    "tts": os.getenv("TTS_BASE", "http://tts-service:8002"),
    "image": os.getenv("IMAGE_BASE", "http://image-service:8003"),
    "moderation": os.getenv("MODERATION_BASE", "http://moderation-service:8006"),
    "video": os.getenv("VIDEO_BASE", "http://video-service:8005"),
    "distribution": os.getenv("DISTRIBUTION_BASE", "http://distribution-service:8007"),
    "analytics": os.getenv("ANALYTICS_BASE", "http://analytics-service:8008"),
}

FLOW_RUNS = Counter('orchestration_flow_runs_total', 'Total flow runs', ['flow', 'status'])
FLOW_DURATION = Histogram('orchestration_flow_duration_seconds', 'Flow duration', ['flow'])
STEP_DURATION = Histogram('orchestration_step_duration_seconds', 'Step duration', ['flow', 'step'])
FLOW_BUDGET_WARNINGS = Counter('orchestration_budget_warnings_total', 'Budget warnings', ['service'])


@task(retries=3, retry_delay_seconds=5)
async def resolve_model_config(run_id: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    try:
        async with get_db_manager().get_session() as session:
            result = await session.execute(
                select(ModelConfiguration).where(ModelConfiguration.is_active == True)
            )
            rows: List[ModelConfiguration] = result.scalars().all()
            by_type: Dict[str, List[ModelConfiguration]] = {}
            for r in rows:
                by_type.setdefault(r.config_type, []).append(r)
            bucket = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest(), 16) % 100
            for config_type, items in by_type.items():
                chosen = next((i for i in items if getattr(i, 'is_default', False)), None)
                for i in items:
                    try:
                        if i.traffic_percentage is not None and bucket < int(i.traffic_percentage):
                            chosen = i
                            break
                    except Exception:
                        continue
                if chosen:
                    cfg[config_type] = {
                        "provider": chosen.provider,
                        "model_name": chosen.model_name,
                        "parameters": chosen.parameters,
                        "version": chosen.version,
                    }
        return cfg
    except Exception:
        return {"story": {"provider": "gpt-4o"}, "tts": {"provider": "elevenlabs"}, "image": {"provider": "dall-e-3"}}


@task(retries=2, retry_delay_seconds=2)
async def check_budget(service: str) -> bool:
    env_key = f"BUDGET_DAILY_{service.upper().replace('-', '_')}"
    try:
        limit = float(os.getenv(env_key, "0"))
    except Exception:
        limit = 0.0
    if limit <= 0:
        return True
    limiter = await get_rate_limiter()
    spent = await limiter.get_daily_cost("orchestrator", service)
    return spent <= limit


@task(retries=2, retry_delay_seconds=2)
async def track_cost(service: str, cost: float) -> None:
    try:
        limiter = await get_rate_limiter()
        await limiter.track_cost("orchestrator", service, cost)
    except Exception:
        pass


@task(retries=3, retry_delay_seconds=5)
async def generate_story(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{BASES['story']}/generate/story", json=payload)
        resp.raise_for_status()
        return resp.json()


@task(retries=3, retry_delay_seconds=5)
async def moderate_text(text: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASES['moderation']}/moderate/text",
            params={"text": text, "level": "medium", "platform": "youtube"},
        )
        resp.raise_for_status()
        return resp.json()


@task(retries=3, retry_delay_seconds=5)
async def synthesize_tts(text: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{BASES['tts']}/synthesize/speech", json={"text": text})
        resp.raise_for_status()
        return resp.json()


@task(retries=3, retry_delay_seconds=5)
async def generate_images(story_id: str, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{BASES['image']}/generate/batch",
            json={"story_id": story_id, "scenes": scenes},
        )
        resp.raise_for_status()
        return resp.json()


@task(retries=3, retry_delay_seconds=5)
async def moderate_image_urls(image_urls: List[str], moderation_level: str = "medium") -> Tuple[bool, List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for url in image_urls:
            try:
                r = await client.get(f"{BASES['image']}{url}")
                r.raise_for_status()
                files = {"file": (os.path.basename(url), r.content, r.headers.get("content-type", "image/png"))}
                mr = await client.post(
                    f"{BASES['moderation']}/moderate/file",
                    params={"content_type": "image", "level": moderation_level, "platform": "youtube"},
                    files=files,
                )
                mr.raise_for_status()
                results.append(mr.json())
            except Exception as e:
                results.append({"error": str(e), "flagged": False, "score": 0})
    approved = all((not item.get("requires_review") and item.get("status", "approved") == "approved") if not item.get("error") else True for item in results)
    return approved, results


@task(retries=3, retry_delay_seconds=5)
async def moderate_audio(audio_id: str, moderation_level: str = "medium") -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        audio_resp = await client.get(f"{BASES['tts']}/audio/{audio_id}")
        audio_resp.raise_for_status()
        files = {"file": (f"{audio_id}.mp3", audio_resp.content, audio_resp.headers.get("content-type", "audio/mpeg"))}
        mr = await client.post(
            f"{BASES['moderation']}/moderate/file",
            params={"content_type": "audio", "level": moderation_level, "platform": "youtube"},
            files=files,
        )
        mr.raise_for_status()
        return mr.json()


@task(retries=3, retry_delay_seconds=5)
async def assemble_video(
    story_id: str,
    title: str,
    description: str,
    image_paths: List[str],
    narration_path: str,
    platform: str = "youtube",
    format_: str = "16:9",
    quality: str = "high",
    image_duration: float = 3.0,
) -> Dict[str, Any]:
    assets = []
    start = 0.0
    for p in image_paths:
        assets.append({"asset_id": os.path.basename(p), "asset_type": "image", "file_path": p, "duration": image_duration, "start_time": start})
        start += image_duration
    assets.append({"asset_id": "narration", "asset_type": "audio", "file_path": narration_path, "volume": 1.0, "start_time": 0.0})
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{BASES['video']}/assemble",
            json={
                "story_id": story_id,
                "title": title,
                "description": description,
                "assets": assets,
                "platform": platform,
                "format": format_,
                "quality": quality,
            },
        )
        resp.raise_for_status()
        return resp.json()


@task(retries=3, retry_delay_seconds=5)
async def distribute_video(video_id: str, title: str, description: str, platforms: List[str], hashtags: List[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "video_id": video_id,
            "platforms": platforms,
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "privacy": "public",
            "idempotency_key": str(uuid.uuid4()),
        }
        resp = await client.post(f"{BASES['distribution']}/upload/batch", json=payload)
        resp.raise_for_status()
        return {"results": resp.json()}


@task(retries=3, retry_delay_seconds=5)
async def trigger_analytics_collection(platforms: List[str]) -> Dict[str, Any]:
    results = {}
    async with httpx.AsyncClient(timeout=60) as client:
        for p in platforms:
            try:
                r = await client.post(f"{BASES['analytics']}/collect/{p}")
                results[p] = {"status": r.status_code, "body": r.json() if r.headers.get("content-type","" ).startswith("application/json") else {}}
            except Exception as e:
                results[p] = {"error": str(e)}
    return results


@flow(name="story_to_video_pipeline")
async def story_to_video_pipeline(
    genre: str = "fantasy",
    theme: str = "adventure",
    target_length: int = 400,
    platforms: List[str] = ["youtube"],
    scene_limit: int = 5,
    image_duration: float = 3.0,
    hashtags: List[str] = ("#story", "#ai"),
    moderation_level: str = "medium",
    platform: str = "youtube",
    format_: str = "16:9",
    quality: str = "high",
) -> Dict[str, Any]:
    logger = get_run_logger()
    flow_name = "story_to_video_pipeline"
    t0 = time.time()
    limiter = await get_rate_limiter()
    if not await limiter.rate_limiter.is_allowed("orchestration:story_to_video", limit=20, window_seconds=3600):
        FLOW_RUNS.labels(flow=flow_name, status="rate_limited").inc()
        return {"status": "rate_limited"}

    run_id = str(uuid.uuid4())
    cfg = await resolve_model_config.submit(run_id).result()

    if not await check_budget.submit("story").result():
        return {"status": "budget_exceeded", "stage": "story"}
    STEP_DURATION.labels(flow_name, "resolve_config").observe(max(time.time()-t0, 0))
    st = time.time()
    story_task = generate_story.submit({"genre": genre, "theme": theme, "target_length": target_length})
    story = await story_task.result()
    await track_cost.submit("story", 0.01)
    story_text = story.get("content", "")
    STEP_DURATION.labels(flow_name, "story").observe(max(time.time()-st, 0))

    mod_text = await moderate_text.submit(story_text).result()
    if mod_text.get("requires_review") or str(mod_text.get("status", "")).lower() in ("flagged", "rejected"):
        logger.warning("Text moderation flagged content. Aborting pipeline.")
        return {"status": "moderation_failed", "stage": "text", "moderation": mod_text}

    if not await check_budget.submit("tts").result():
        return {"status": "budget_exceeded", "stage": "tts"}

    sentences = [s.strip() for s in story_text.split(".") if len(s.strip()) > 10][:scene_limit]
    scenes = [{"scene_number": i+1, "prompt": f"{s}, cinematic lighting, high quality"} for i, s in enumerate(sentences)]
    if not await check_budget.submit("image").result():
        return {"status": "budget_exceeded", "stage": "image"}

    st = time.time()
    tts_fut = synthesize_tts.submit(story_text)
    images_fut = generate_images.submit(story["id"], scenes)

    tts = await tts_fut.result()
    await track_cost.submit("tts", 0.02)
    images = await images_fut.result()
    await track_cost.submit("image", 0.03)
    STEP_DURATION.labels(flow_name, "tts+images").observe(max(time.time()-st, 0))

    narration_path = tts.get("file_path")
    audio_id = tts.get("audio_id")
    image_paths = [img["file_path"] for img in images.get("images", [])]
    image_urls = [img.get("file_url") for img in images.get("images", []) if img.get("file_url")]

    st = time.time()
    img_mod_fut = moderate_image_urls.submit(image_urls, moderation_level)
    aud_mod_fut = moderate_audio.submit(audio_id, moderation_level)
    approved_images, image_mod_details = await img_mod_fut.result()
    audio_mod = await aud_mod_fut.result()
    STEP_DURATION.labels(flow_name, "asset_moderation").observe(max(time.time()-st, 0))
    if not approved_images:
        logger.warning("Image moderation flagged content. Aborting pipeline.")
        return {"status": "moderation_failed", "stage": "images", "details": image_mod_details}
    if audio_mod.get("requires_review") or str(audio_mod.get("status", "")).lower() in ("flagged", "rejected"):
        logger.warning("Audio moderation flagged content. Aborting pipeline.")
        return {"status": "moderation_failed", "stage": "audio", "moderation": audio_mod}

    if not await check_budget.submit("video").result():
        return {"status": "budget_exceeded", "stage": "video"}
    st = time.time()
    video = await assemble_video.submit(
        story["id"], "AI Story Video", "Generated by pipeline", image_paths, narration_path, platform, format_, quality, image_duration
    ).result()
    if str(video.get("status", "")).lower() != "completed":
        FLOW_RUNS.labels(flow=flow_name, status="video_failed").inc()
        FLOW_DURATION.labels(flow=flow_name).observe(max(time.time()-t0, 0))
        return {"status": "video_failed", "video": video}
    await track_cost.submit("video", 0.05)
    STEP_DURATION.labels(flow_name, "video").observe(max(time.time()-st, 0))

    st = time.time()
    dist_fut = distribute_video.submit(video.get("video_id"), "AI Story Video", "Generated by pipeline", platforms, list(hashtags))
    analytics_fut = trigger_analytics_collection.submit(platforms)
    dist = await dist_fut.result()
    analytics = await analytics_fut.result()
    STEP_DURATION.labels(flow_name, "dist+analytics").observe(max(time.time()-st, 0))
    FLOW_RUNS.labels(flow=flow_name, status="completed").inc()
    FLOW_DURATION.labels(flow=flow_name).observe(max(time.time()-t0, 0))

    return {"status": "completed", "story": story, "tts": tts, "images": images, "video": video, "distribution": dist, "analytics": analytics}


@flow(name="content_generation_flow")
async def content_generation_flow(genre: str = "fantasy", theme: str = "adventure", target_length: int = 400) -> Dict[str, Any]:
    story = await generate_story.submit({"genre": genre, "theme": theme, "target_length": target_length}).result()
    tts = await synthesize_tts.submit(story.get("content", "")).result()
    sentences = [s.strip() for s in story.get("content", "").split(".") if len(s.strip()) > 10][:5]
    scenes = [{"scene_number": i+1, "prompt": f"{s}, cinematic lighting, high quality"} for i, s in enumerate(sentences)]
    images = await generate_images.submit(story["id"], scenes).result()
    return {"story": story, "tts": tts, "images": images}


@flow(name="video_assembly_flow")
async def video_assembly_flow(story_id: str, title: str, image_paths: List[str], narration_path: str) -> Dict[str, Any]:
    video = await assemble_video.submit(story_id, title, "", image_paths, narration_path).result()
    return {"video": video}


@flow(name="distribution_flow")
async def distribution_flow(video_id: str, title: str, description: str, platforms: List[str]) -> Dict[str, Any]:
    dist = await distribute_video.submit(video_id, title, description, platforms).result()
    return {"distribution": dist}


@flow(name="analytics_collection_flow")
async def analytics_collection_flow(platforms: List[str] = ["youtube"]) -> Dict[str, Any]:
    return await trigger_analytics_collection.submit(platforms).result()
