from __future__ import annotations

from typing import Dict, Any
from datetime import datetime
import os
import random

from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from sqlalchemy import select

from shared.models import PlatformUpload, PlatformType
from shared.database import get_db_manager
from shared.rate_limiter import get_rate_limiter
from shared.retry import retry_api_call, APIError
from .base_collector import BaseCollector


YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class YouTubeCollector(BaseCollector):
    platform = PlatformType.YOUTUBE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.creds = None
        creds_json = os.getenv("YOUTUBE_ANALYTICS_CREDENTIALS")
        if creds_json:
            try:
                self.creds = Credentials.from_service_account_info(
                    eval(creds_json), scopes=YOUTUBE_SCOPES
                )
            except Exception:
                # If invalid creds, keep None and fallback to mock
                self.creds = None

    @retry_api_call(max_retries=3)
    async def _fetch_real(self, upload: PlatformUpload) -> Dict[str, Any]:
        if not self.creds:
            raise APIError("YouTube credentials not configured")
        limiter = await get_rate_limiter()
        if not await limiter.rate_limiter.is_allowed("youtube_analytics_api", 1000, 3600):
            raise APIError("Rate limited")

        # Note: googleapiclient is sync; run in thread if needed. For brevity, call directly.
        service = build('youtubeAnalytics', 'v2', credentials=self.creds, cache_discovery=False)
        # Minimal sample: use reports().query for basic views/likes; real implementation would set ids and metrics
        # This is a placeholder; platforms often need channel ID and video ID
        # For safety in this stub, fallback to mock if missing video ID
        if not upload.platform_video_id:
            raise APIError("Missing platform_video_id")

        # Placeholder: return empty to avoid real API calls in dev
        # Replace with service.reports().query(...).execute()
        return {}

    async def fetch_metrics(self, upload: PlatformUpload) -> Dict[str, Any]:
        # Try real API if creds present; else mock
        if self.creds:
            try:
                data = await self._fetch_real(upload)
                if data:
                    now = datetime.utcnow()
                    # Map fields from data to our schema; placeholder mapping
                    return {
                        "views": data.get("views", 0),
                        "likes": data.get("likes", 0),
                        "comments": data.get("comments", 0),
                        "shares": data.get("shares", 0),
                        "completion_rate": data.get("completionRate"),
                        "ctr": data.get("ctr"),
                        "engagement_rate": data.get("engagementRate"),
                        "ad_revenue": data.get("adRevenue", 0.0),
                        "creator_fund_revenue": 0.0,
                        "collected_at": now,
                        "raw": data,
                    }
            except Exception:
                # Fall back to mock
                pass
        # Mock data
        if not upload.platform_video_id:
            return {}
        now = datetime.utcnow()
        return {
            "views": random.randint(0, 1000),
            "likes": random.randint(0, 200),
            "comments": random.randint(0, 80),
            "shares": random.randint(0, 50),
            "completion_rate": round(random.uniform(0.2, 0.9), 2),
            "ctr": round(random.uniform(0.01, 0.15), 3),
            "engagement_rate": round(random.uniform(0.02, 0.3), 3),
            "ad_revenue": round(random.uniform(0, 10), 2),
            "creator_fund_revenue": 0.0,
            "collected_at": now,
            "raw": {"mock": True, "collected_at": now.isoformat()},
        }
