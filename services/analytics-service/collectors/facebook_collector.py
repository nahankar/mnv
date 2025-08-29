from __future__ import annotations

from typing import Dict, Any
from datetime import datetime
import random

from shared.models import PlatformUpload, PlatformType
from .base_collector import BaseCollector


class FacebookCollector(BaseCollector):
    platform = PlatformType.FACEBOOK

    async def fetch_metrics(self, upload: PlatformUpload) -> Dict[str, Any]:
        if not upload.platform_video_id:
            return {}
        now = datetime.utcnow()
        return {
            "views": random.randint(0, 1500),
            "likes": random.randint(0, 300),
            "comments": random.randint(0, 100),
            "shares": random.randint(0, 120),
            "completion_rate": round(random.uniform(0.25, 0.9), 2),
            "ctr": round(random.uniform(0.005, 0.1), 3),
            "engagement_rate": round(random.uniform(0.02, 0.35), 3),
            "ad_revenue": round(random.uniform(0, 5), 2),
            "creator_fund_revenue": 0.0,
            "collected_at": now,
            "raw": {"mock": True, "collected_at": now.isoformat()},
        }
