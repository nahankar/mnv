from __future__ import annotations

from typing import Dict, Any
from datetime import datetime
import random

from shared.models import PlatformUpload, PlatformType
from .base_collector import BaseCollector


class InstagramCollector(BaseCollector):
    platform = PlatformType.INSTAGRAM

    async def fetch_metrics(self, upload: PlatformUpload) -> Dict[str, Any]:
        if not upload.platform_video_id:
            return {}
        now = datetime.utcnow()
        return {
            "views": random.randint(0, 2000),
            "likes": random.randint(0, 500),
            "comments": random.randint(0, 120),
            "shares": random.randint(0, 80),
            "completion_rate": round(random.uniform(0.2, 0.95), 2),
            "ctr": round(random.uniform(0.01, 0.2), 3),
            "engagement_rate": round(random.uniform(0.02, 0.4), 3),
            "ad_revenue": 0.0,
            "creator_fund_revenue": 0.0,
            "collected_at": now,
            "raw": {"mock": True, "collected_at": now.isoformat()},
        }
