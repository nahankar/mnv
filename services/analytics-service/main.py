"""
Analytics Tracking Service

Collects engagement metrics from platforms and stores them in AnalyticsData,
links to PlatformUpload, and exposes aggregation/query endpoints.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import select, text

from shared.database import get_db_manager, DatabaseManager
from shared.logging import get_logger
from shared.middleware import CorrelationMiddleware
from shared.models import AnalyticsData, PlatformUpload, PlatformType

# Metrics
REQUEST_COUNT = Counter('analytics_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('analytics_service_request_duration_seconds', 'Request duration')
INGEST_COUNT = Counter('analytics_ingest_total', 'Ingested analytics rows', ['platform'])

app = FastAPI(title="Analytics Service", version="1.0.0")
logger = get_logger(__name__)

# Middleware
app.add_middleware(CorrelationMiddleware)


class AnalyticsIngestRequest(BaseModel):
    platform_upload_id: str = Field(...)
    platform: PlatformType
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    completion_rate: Optional[float] = None
    ctr: Optional[float] = None
    engagement_rate: Optional[float] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    raw: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsQueryResponse(BaseModel):
    video_id: Optional[str]
    aggregates: Dict[str, Any]
    points: List[Dict[str, Any]]


@app.on_event("startup")
async def startup():
    # Ensure DB connection is initialized
    db = get_db_manager()
    await db.initialize()
    logger.info("Analytics service started")


@app.on_event("shutdown")
async def shutdown():
    await get_db_manager().close()
    logger.info("Analytics service shutdown complete")


@app.get("/health")
async def health(deep: bool = False):
    status = {"status": "healthy", "service": "analytics-service"}
    if deep:
        try:
            async with get_db_manager().get_session() as session:
                await session.execute(text("SELECT 1"))
            status["database"] = "connected"
        except Exception as e:
            status["status"] = "unhealthy"
            status["database"] = f"error: {str(e)}"
    return status


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/analytics/ingest")
async def ingest_analytics(payload: AnalyticsIngestRequest):
    try:
        async with get_db_manager().get_session() as session:
            # Ensure upload exists
            result = await session.execute(
                select(PlatformUpload).filter(PlatformUpload.id == payload.platform_upload_id)
            )
            upload = result.scalar_one_or_none()
            if not upload:
                raise HTTPException(status_code=404, detail="Platform upload not found")
            
            analytics_row = AnalyticsData(
                platform_upload_id=upload.id,
                views=payload.views,
                likes=payload.likes,
                comments=payload.comments,
                shares=payload.shares,
                completion_rate=payload.completion_rate,
                ctr=payload.ctr,
                engagement_rate=payload.engagement_rate,
                data_collected_at=payload.collected_at,
                raw_analytics=payload.raw
            )
            session.add(analytics_row)
            await session.commit()
            INGEST_COUNT.labels(platform=payload.platform.value).inc()
            return {"status": "ok", "id": str(analytics_row.id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Ingest failed")


@app.get("/analytics/video/{video_id}", response_model=AnalyticsQueryResponse)
async def get_video_analytics(
    video_id: str,
    since_hours: int = Query(168, ge=1, le=8760)
):
    try:
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        async with get_db_manager().get_session() as session:
            # Join uploads -> analytics
            from sqlalchemy import join
            j = join(PlatformUpload, AnalyticsData, PlatformUpload.id == AnalyticsData.platform_upload_id)
            result = await session.execute(
                select(
                    PlatformUpload.video_id,
                    AnalyticsData.views,
                    AnalyticsData.likes,
                    AnalyticsData.comments,
                    AnalyticsData.shares,
                    AnalyticsData.completion_rate,
                    AnalyticsData.ctr,
                    AnalyticsData.engagement_rate,
                    AnalyticsData.data_collected_at
                ).select_from(j)
                 .where(PlatformUpload.video_id == video_id)
                 .where(AnalyticsData.data_collected_at >= cutoff)
            )
            rows = result.all()
            
            # Aggregate
            agg = {
                "views": 0,
                "likes": 0,
                "comments": 0,
                "shares": 0,
                "avg_completion_rate": None,
                "avg_ctr": None,
                "avg_engagement_rate": None
            }
            points = []
            crs, ctrs, ers = [], [], []
            for r in rows:
                _, views, likes, comments, shares, cr, ctr, er, t = r
                agg["views"] += views or 0
                agg["likes"] += likes or 0
                agg["comments"] += comments or 0
                agg["shares"] += shares or 0
                if cr is not None: crs.append(cr)
                if ctr is not None: ctrs.append(ctr)
                if er is not None: ers.append(er)
                points.append({
                    "timestamp": t.isoformat(),
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "completion_rate": cr,
                    "ctr": ctr,
                    "engagement_rate": er
                })
            
            def avg(v):
                return (sum(v) / len(v)) if v else None
            agg["avg_completion_rate"] = avg(crs)
            agg["avg_ctr"] = avg(ctrs)
            agg["avg_engagement_rate"] = avg(ers)
            
            return AnalyticsQueryResponse(
                video_id=video_id,
                aggregates=agg,
                points=points
            )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")


@app.get("/analytics/platform/{platform}")
async def platform_aggregate(platform: PlatformType, since_hours: int = Query(24, ge=1, le=8760)):
    try:
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        async with get_db_manager().get_session() as session:
            from sqlalchemy import join
            j = join(PlatformUpload, AnalyticsData, PlatformUpload.id == AnalyticsData.platform_upload_id)
            result = await session.execute(
                select(
                    AnalyticsData.views,
                    AnalyticsData.likes,
                    AnalyticsData.comments,
                    AnalyticsData.shares
                ).select_from(j)
                 .where(PlatformUpload.platform == platform)
                 .where(AnalyticsData.data_collected_at >= cutoff)
            )
            rows = result.all()
            agg = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            for views, likes, comments, shares in rows:
                agg["views"] += views or 0
                agg["likes"] += likes or 0
                agg["comments"] += comments or 0
                agg["shares"] += shares or 0
            return {"platform": platform, "since_hours": since_hours, "aggregates": agg}
    except Exception as e:
        logger.error(f"Platform aggregate failed: {e}")
        raise HTTPException(status_code=500, detail="Platform aggregate failed")


@app.post("/collect/youtube")
async def manual_collect_youtube():
    from .collectors.youtube_collector import YouTubeCollector
    from sqlalchemy import select
    try:
        collector = YouTubeCollector()
        async with get_db_manager().get_session() as session:
            res = await session.execute(select(PlatformUpload).where(PlatformUpload.platform == PlatformType.YOUTUBE))
            uploads = list(res.scalars().all())
        count = await collector.collect_for_video_uploads(uploads)
        return {"status": "ok", "collected": count, "uploads": len(uploads)}
    except Exception as e:
        logger.error(f"Manual collect failed: {e}")
        raise HTTPException(status_code=500, detail="Manual collect failed")


@app.post("/collect/{platform}")
async def manual_collect_platform(platform: PlatformType):
    from .collectors.youtube_collector import YouTubeCollector
    from .collectors.instagram_collector import InstagramCollector
    from .collectors.tiktok_collector import TikTokCollector
    from .collectors.facebook_collector import FacebookCollector
    from sqlalchemy import select
    try:
        collector_map = {
            PlatformType.YOUTUBE: YouTubeCollector,
            PlatformType.INSTAGRAM: InstagramCollector,
            PlatformType.TIKTOK: TikTokCollector,
            PlatformType.FACEBOOK: FacebookCollector,
        }
        collector_cls = collector_map.get(platform)
        if not collector_cls:
            raise HTTPException(status_code=400, detail="Unsupported platform")
        collector = collector_cls()
        async with get_db_manager().get_session() as session:
            res = await session.execute(select(PlatformUpload).where(PlatformUpload.platform == platform))
            uploads = list(res.scalars().all())
        count = await collector.collect_for_video_uploads(uploads)
        return {"status": "ok", "platform": platform.value, "collected": count, "uploads": len(uploads)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual collect failed: {e}")
        raise HTTPException(status_code=500, detail="Manual collect failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
