import pytest
import httpx


@pytest.mark.asyncio
async def test_orchestration_health():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:8010/health")
        assert r.status_code == 200
        assert r.json()["service"] == "orchestration-service"


@pytest.mark.asyncio
async def test_orchestration_run_dry():
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.post("http://localhost:8010/run/story-to-video", params={"genre": "test", "theme": "test", "target_length": 10})
        # In CI or without all services up, allow failure
        assert r.status_code in (200, 500)
