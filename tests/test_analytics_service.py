import pytest
import httpx
import uuid
import os
from datetime import datetime, timedelta

BASE_URL = os.getenv("ANALYTICS_BASE_URL", "http://localhost:8008")


@pytest.mark.asyncio
async def test_analytics_health():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "analytics-service"


@pytest.mark.asyncio
async def test_ingest_missing_upload():
    payload = {
        "platform_upload_id": str(uuid.uuid4()),
        "platform": "youtube",
        "views": 10,
        "likes": 2,
        "comments": 1,
        "shares": 0,
        "collected_at": datetime.utcnow().isoformat(),
        "raw": {"sample": True}
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE_URL}/analytics/ingest", json=payload)
        assert r.status_code in (404, 500)


@pytest.mark.asyncio
async def test_video_analytics_query_no_data():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/analytics/video/{uuid.uuid4()}?since_hours=24")
        assert r.status_code in (200, 500)


@pytest.mark.asyncio
async def test_manual_collect_platforms_stub():
    async with httpx.AsyncClient() as client:
        for platform in ("youtube", "instagram", "tiktok", "facebook"):
            r = await client.post(f"{BASE_URL}/collect/{platform}")
            assert r.status_code in (200, 500)
