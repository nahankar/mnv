from __future__ import annotations

import abc
from typing import List, Dict, Any
from datetime import datetime

from shared.database import DatabaseManager, get_db_manager
from shared.models import PlatformUpload, AnalyticsData, PlatformType


class BaseCollector(abc.ABC):
    """Abstract collector for platform analytics."""

    platform: PlatformType

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    @abc.abstractmethod
    async def fetch_metrics(self, upload: PlatformUpload) -> Dict[str, Any]:
        """Fetch metrics from platform API for a given upload.
        Must return a dict with fields compatible with AnalyticsData and 'raw'.
        """
        raise NotImplementedError

    async def collect_for_upload(self, upload: PlatformUpload) -> AnalyticsData | None:
        metrics = await self.fetch_metrics(upload)
        if not metrics:
            return None
        async with self.db.get_session() as session:
            row = AnalyticsData(
                platform_upload_id=upload.id,
                views=metrics.get("views", 0),
                likes=metrics.get("likes", 0),
                comments=metrics.get("comments", 0),
                shares=metrics.get("shares", 0),
                completion_rate=metrics.get("completion_rate"),
                ctr=metrics.get("ctr"),
                engagement_rate=metrics.get("engagement_rate"),
                ad_revenue=metrics.get("ad_revenue", 0.0),
                creator_fund_revenue=metrics.get("creator_fund_revenue", 0.0),
                data_collected_at=metrics.get("collected_at", datetime.utcnow()),
                raw_analytics=metrics.get("raw", {}),
            )
            session.add(row)
            await session.commit()
            return row

    async def collect_for_video_uploads(self, uploads: List[PlatformUpload]) -> int:
        count = 0
        for u in uploads:
            if await self.collect_for_upload(u):
                count += 1
        return count
