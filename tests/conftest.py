"""Test configuration and fixtures"""
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.database import Base
from shared.models import *


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    # Use in-memory SQLite for tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine):
    """Create test database session"""
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def sample_story(test_session):
    """Create a sample story for testing"""
    story = Story(
        title="Test Story",
        content="This is a test story content.",
        genre="fantasy",
        theme="adventure",
        target_length=100,
        actual_length=95,
        llm_provider="openai",
        model_name="gpt-4",
        generation_parameters={"temperature": 0.7},
        generation_cost=0.05
    )
    test_session.add(story)
    await test_session.commit()
    await test_session.refresh(story)
    return story


@pytest.fixture
async def sample_media_asset(test_session, sample_story):
    """Create a sample media asset for testing"""
    asset = MediaAsset(
        story_id=sample_story.id,
        asset_type=MediaType.AUDIO,
        file_path="/test/audio.mp3",
        file_size=1024000,
        duration=60.5,
        metadata={"format": "mp3", "bitrate": 128},
        provider="elevenlabs",
        model_name="eleven_monolingual_v1",
        generation_parameters={"voice": "adam"},
        generation_cost=0.10,
        prompt_used="Generate audio for this story"
    )
    test_session.add(asset)
    await test_session.commit()
    await test_session.refresh(asset)
    return asset


@pytest.fixture
async def sample_video(test_session, sample_story):
    """Create a sample video for testing"""
    video = Video(
        story_id=sample_story.id,
        title="Test Video",
        description="Test video description",
        file_path="/test/video.mp4",
        file_size=5000000,
        duration=120.0,
        format_type="16:9",
        resolution="1920x1080",
        target_platforms=["youtube", "instagram"],
        assembly_parameters={"transitions": True},
        generation_cost=0.25
    )
    test_session.add(video)
    await test_session.commit()
    await test_session.refresh(video)
    return video