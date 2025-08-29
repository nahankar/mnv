import os
from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery(
    "analytics",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "collect-youtube-hourly": {
            "task": "tasks.collect_youtube_hourly",
            "schedule": 60 * 60,
        },
        "collect-instagram-hourly": {
            "task": "tasks.collect_instagram_hourly",
            "schedule": 60 * 60,
        },
        "collect-tiktok-hourly": {
            "task": "tasks.collect_tiktok_hourly",
            "schedule": 60 * 60,
        },
        "collect-facebook-hourly": {
            "task": "tasks.collect_facebook_hourly",
            "schedule": 60 * 60,
        },
    },
)
