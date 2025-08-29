import base64
import os
import pytest
import httpx


@pytest.mark.asyncio
async def test_validate_quality_base64():
    """Validate quality endpoint with a tiny base64 PNG (detailed=False)."""
    # 1x1 transparent PNG
    tiny_png = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAOQjX1EAAAAASUVORK5CYII="
    )
    data_url = "data:image/png;base64," + tiny_png.decode()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8003/validate/quality",
            json={"image_source": data_url, "detailed": False},
            timeout=10.0,
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "metrics" in body
    metrics = body["metrics"]
    # Ensure metrics are present (numbers, not None)
    for key in [
        "resolution_score",
        "sharpness_score",
        "brightness_score",
        "contrast_score",
        "overall_score",
    ]:
        assert key in metrics
        assert metrics[key] is not None
    assert "quality_grade" in body
    assert body["quality_grade"] in {"Excellent", "Good", "Fair", "Poor", "Very Poor"}


