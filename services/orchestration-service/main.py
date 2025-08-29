import os
import asyncio
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from flows import (
    story_to_video_pipeline,
    content_generation_flow,
    video_assembly_flow,
    distribution_flow,
    analytics_collection_flow,
)
from shared.database import get_db_manager
from shared.models import ModelConfiguration
from shared.schemas import ModelConfigurationRequest
from shared.rate_limiter import get_rate_limiter

app = FastAPI(title="Orchestration Service", version="1.0.0")

LAST_RUNS = deque(maxlen=100)


class BudgetUpdate(BaseModel):
    budgets: Dict[str, float]


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestration-service"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/run/story-to-video")
async def run_story_to_video(genre: str = "fantasy", theme: str = "adventure", target_length: int = 400):
    run_id = f"run-{datetime.utcnow().timestamp()}"
    try:
        result = await story_to_video_pipeline(genre=genre, theme=theme, target_length=target_length)
        entry = {"id": run_id, "flow": "story_to_video_pipeline", "params": {"genre": genre, "theme": theme, "target_length": target_length}, "status": result.get("status", "unknown"), "created_at": datetime.utcnow().isoformat(), "result": result}
        LAST_RUNS.appendleft(entry)
        return {"status": "completed", "result": result, "run_id": run_id}
    except Exception as e:
        entry = {"id": run_id, "flow": "story_to_video_pipeline", "params": {"genre": genre, "theme": theme, "target_length": target_length}, "status": "error", "created_at": datetime.utcnow().isoformat(), "error": str(e)}
        LAST_RUNS.appendleft(entry)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/content-generation")
async def run_content_generation(genre: str = "fantasy", theme: str = "adventure", target_length: int = 400):
    try:
        result = await content_generation_flow(genre=genre, theme=theme, target_length=target_length)
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/video-assembly")
async def run_video_assembly(story_id: str, title: str, narration_path: str, image_paths: str):
    try:
        paths = [p for p in image_paths.split(",") if p]
        result = await video_assembly_flow(story_id=story_id, title=title, image_paths=paths, narration_path=narration_path)
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/distribution")
async def run_distribution(video_id: str, title: str, description: str = "", platforms: str = "youtube"):
    try:
        p = [x.strip() for x in platforms.split(",") if x.strip()]
        result = await distribution_flow(video_id=video_id, title=title, description=description, platforms=p)
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/analytics-collection")
async def run_analytics_collection(platforms: str = "youtube"):
    try:
        p = [x.strip() for x in platforms.split(",") if x.strip()]
        result = await analytics_collection_flow(platforms=p)
        entry = {"id": f"run-{datetime.utcnow().timestamp()}", "flow": "analytics_collection_flow", "params": {"platforms": p}, "status": "completed", "created_at": datetime.utcnow().isoformat(), "result": result}
        LAST_RUNS.appendleft(entry)
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

# --- Model Configuration CRUD ---

@app.get("/model-configs")
async def list_model_configs(config_type: Optional[str] = None):
    async with get_db_manager().get_session() as session:
        query = session.query(ModelConfiguration)
        if config_type:
            query = query.filter(ModelConfiguration.config_type == config_type)
        results = await session.execute(query)
        items = results.scalars().all()
        return {"items": [
            {
                "id": str(it.id),
                "name": it.name,
                "description": it.description,
                "config_type": it.config_type,
                "provider": it.provider,
                "model_name": it.model_name,
                "parameters": it.parameters,
                "cost_per_unit": it.cost_per_unit,
                "performance_metrics": it.performance_metrics,
                "version": it.version,
                "is_active": it.is_active,
                "is_default": it.is_default,
                "ab_test_group": it.ab_test_group,
                "traffic_percentage": it.traffic_percentage,
                "created_at": it.created_at,
                "updated_at": it.updated_at,
            }
            for it in items
        ]}


@app.post("/model-configs")
async def create_model_config(req: ModelConfigurationRequest):
    async with get_db_manager().get_session() as session:
        mc = ModelConfiguration(
            name=req.name,
            description=req.description,
            config_type=req.config_type,
            provider=req.provider,
            model_name=req.model_name,
            parameters=req.parameters,
            cost_per_unit=req.cost_per_unit,
            performance_metrics=req.performance_metrics,
            version=req.version,
            is_active=req.is_active,
            is_default=req.is_default,
            ab_test_group=req.ab_test_group,
            traffic_percentage=req.traffic_percentage,
        )
        session.add(mc)
        await session.commit()
        await session.refresh(mc)
        return {"id": str(mc.id)}


@app.put("/model-configs/{config_id}")
async def update_model_config(config_id: str, req: ModelConfigurationRequest):
    async with get_db_manager().get_session() as session:
        result = await session.execute(session.query(ModelConfiguration).filter(ModelConfiguration.id == config_id))
        mc = result.scalar_one_or_none()
        if not mc:
            raise HTTPException(status_code=404, detail="Not found")
        for field in req.model_fields.keys():
            setattr(mc, field, getattr(req, field))
        await session.commit()
        return {"status": "updated"}


@app.delete("/model-configs/{config_id}")
async def delete_model_config(config_id: str):
    async with get_db_manager().get_session() as session:
        result = await session.execute(session.query(ModelConfiguration).filter(ModelConfiguration.id == config_id))
        mc = result.scalar_one_or_none()
        if not mc:
            raise HTTPException(status_code=404, detail="Not found")
        await session.delete(mc)
        await session.commit()
        return {"status": "deleted"}


# --- Budgets and Costs ---

@app.get("/budgets")
async def get_budgets():
    limiter = await get_rate_limiter()
    rc = limiter.rate_limiter.redis_client
    budgets: Dict[str, float] = {}
    if rc:
        for service in ["story_generation", "tts", "image_generation", "video_assembly", "distribution", "analytics_collection", "moderation"]:
            key = f"budget:{service}:daily_limit"
            val = await rc.get(key)
            if val:
                budgets[service] = float(val)
    return {"budgets": budgets}


@app.post("/budgets")
async def set_budgets(payload: BudgetUpdate):
    limiter = await get_rate_limiter()
    rc = limiter.rate_limiter.redis_client
    if not rc:
        raise HTTPException(status_code=500, detail="Redis not available")
    for service, limit in payload.budgets.items():
        await rc.set(f"budget:{service}:daily_limit", limit)
    return {"status": "ok"}


@app.get("/costs")
async def get_costs(user_id: str = "admin"):
    limiter = await get_rate_limiter()
    services = ["story_generation", "tts", "image_generation", "video_assembly", "distribution", "analytics_collection", "moderation"]
    costs = {}
    for svc in services:
        costs[svc] = await limiter.get_daily_cost(user_id, svc)
    return {"user_id": user_id, "costs": costs}


# --- Run history ---

@app.get("/runs")
async def list_runs():
    return {"items": list(LAST_RUNS)}


@app.get("/runs/stream")
async def runs_stream():
    async def event_generator():
        last_len = 0
        while True:
            try:
                await asyncio.sleep(2)
                cur = list(LAST_RUNS)
                if len(cur) != last_len:
                    last_len = len(cur)
                    yield f"data: {cur[:10]}\n\n"
            except asyncio.CancelledError:
                break
    return StreamingResponse(event_generator(), media_type="text/event-stream")
