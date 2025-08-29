from __future__ import annotations

from typing import Dict, Any
from datetime import datetime
import random

from shared.models import PlatformUpload, PlatformType
from .base_collector import BaseCollector


class TikTokCollector(BaseCollector):
    platform = PlatformType.TIKTOK

    async def fetch_metrics(self, upload: PlatformUpload) -> Dict[str, Any]:
        if not upload.platform_video_id:
            return {}
        now = datetime.utcnow()
        return {
            "views": random.randint(0, 5000),
            "likes": random.randint(0, 800),
            "comments": random.randint(0, 200),
            "shares": random.randint(0, 300),
            "completion_rate": round(random.uniform(0.3, 0.98), 2),
            "ctr": round(random.uniform(0.005, 0.12), 3),
            "engagement_rate": round(random.uniform(0.03, 0.5), 3),
            "ad_revenue": 0.0,
            "creator_fund_revenue": round(random.uniform(0, 8), 2),
            "collected_at": now,
            "raw": {"mock": True, "collected_at": now.isoformat()},
        }
