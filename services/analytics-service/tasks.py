import asyncio
from datetime import datetime, timedelta
from typing import List

from sqlalchemy import select

from .celery_app import celery_app
from .collectors.youtube_collector import YouTubeCollector
from .collectors.instagram_collector import InstagramCollector
from .collectors.tiktok_collector import TikTokCollector
from .collectors.facebook_collector import FacebookCollector
from shared.database import get_db_manager
from shared.models import PlatformUpload, PlatformType


async def _get_uploads(platform: PlatformType) -> List[PlatformUpload]:
    async with get_db_manager().get_session() as session:
        result = await session.execute(
            select(PlatformUpload).where(PlatformUpload.platform == platform)
        )
        return list(result.scalars().all())


@celery_app.task(name="tasks.collect_youtube_hourly")
def collect_youtube_hourly():
    async def runner():
        uploads = await _get_uploads(PlatformType.YOUTUBE)
        collector = YouTubeCollector()
        await collector.db.initialize()
        try:
            count = await collector.collect_for_video_uploads(uploads)
            return {"platform": "youtube", "collected": count, "uploads": len(uploads)}
        finally:
            await collector.db.close()
    return asyncio.run(runner())


@celery_app.task(name="tasks.collect_instagram_hourly")
def collect_instagram_hourly():
    async def runner():
        uploads = await _get_uploads(PlatformType.INSTAGRAM)
        collector = InstagramCollector()
        await collector.db.initialize()
        try:
            count = await collector.collect_for_video_uploads(uploads)
            return {"platform": "instagram", "collected": count, "uploads": len(uploads)}
        finally:
            await collector.db.close()
    return asyncio.run(runner())


@celery_app.task(name="tasks.collect_tiktok_hourly")
def collect_tiktok_hourly():
    async def runner():
        uploads = await _get_uploads(PlatformType.TIKTOK)
        collector = TikTokCollector()
        await collector.db.initialize()
        try:
            count = await collector.collect_for_video_uploads(uploads)
            return {"platform": "tiktok", "collected": count, "uploads": len(uploads)}
        finally:
            await collector.db.close()
    return asyncio.run(runner())


@celery_app.task(name="tasks.collect_facebook_hourly")
def collect_facebook_hourly():
    async def runner():
        uploads = await _get_uploads(PlatformType.FACEBOOK)
        collector = FacebookCollector()
        await collector.db.initialize()
        try:
            count = await collector.collect_for_video_uploads(uploads)
            return {"platform": "facebook", "collected": count, "uploads": len(uploads)}
        finally:
            await collector.db.close()
    return asyncio.run(runner())
