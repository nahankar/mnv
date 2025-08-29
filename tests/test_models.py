"""Tests for SQLAlchemy models"""
import pytest
from datetime import datetime
from sqlalchemy import select
import uuid

from shared.models import (
    Story, MediaAsset, Video, PlatformUpload, AnalyticsData, ModelConfiguration,
    ContentStatus, PlatformType, MediaType
)


class TestStoryModel:
    """Tests for Story model"""
    
    async def test_create_story(self, test_session):
        """Test creating a story"""
        story = Story(
            title="Test Story",
            content="This is a test story.",
            genre="sci-fi",
            theme="space exploration",
            target_length=150,
            llm_provider="openai"
        )
        
        test_session.add(story)
        await test_session.commit()
        await test_session.refresh(story)
        
        assert story.id is not None
        assert story.title == "Test Story"
        assert story.content == "This is a test story."
        assert story.genre == "sci-fi"
        assert story.status == ContentStatus.PENDING
        assert story.created_at is not None
    
    async def test_story_relationships(self, test_session, sample_story):
        """Test story relationships with media assets and videos"""
        # Add media asset
        asset = MediaAsset(
            story_id=sample_story.id,
            asset_type=MediaType.IMAGE,
            file_path="/test/image.jpg",
            provider="dall-e"
        )
        test_session.add(asset)
        
        # Add video
        video = Video(
            story_id=sample_story.id,
            title="Test Video",
            file_path="/test/video.mp4",
            duration=60.0,
            format_type="16:9"
        )
        test_session.add(video)
        
        await test_session.commit()
        
        # Refresh story to load relationships
        await test_session.refresh(sample_story)
        
        assert len(sample_story.media_assets) == 1
        assert len(sample_story.videos) == 1
        assert sample_story.media_assets[0].asset_type == MediaType.IMAGE
        assert sample_story.videos[0].title == "Test Video"


class TestMediaAssetModel:
    """Tests for MediaAsset model"""
    
    async def test_create_media_asset(self, test_session, sample_story):
        """Test creating a media asset"""
        asset = MediaAsset(
            story_id=sample_story.id,
            asset_type=MediaType.AUDIO,
            file_path="/test/audio.wav",
            file_size=2048000,
            duration=90.5,
            metadata={"format": "wav", "sample_rate": 44100},
            provider="openai",
            model_name="tts-1",
            generation_cost=0.15
        )
        
        test_session.add(asset)
        await test_session.commit()
        await test_session.refresh(asset)
        
        assert asset.id is not None
        assert asset.story_id == sample_story.id
        assert asset.asset_type == MediaType.AUDIO
        assert asset.file_size == 2048000
        assert asset.duration == 90.5
        assert asset.metadata["format"] == "wav"
        assert asset.generation_cost == 0.15
    
    async def test_media_asset_story_relationship(self, test_session, sample_media_asset):
        """Test media asset relationship with story"""
        await test_session.refresh(sample_media_asset, ["story"])
        
        assert sample_media_asset.story is not None
        assert sample_media_asset.story.title == "Test Story"


class TestVideoModel:
    """Tests for Video model"""
    
    async def test_create_video(self, test_session, sample_story):
        """Test creating a video"""
        video = Video(
            story_id=sample_story.id,
            title="My Test Video",
            description="A video for testing",
            file_path="/test/my_video.mp4",
            file_size=10000000,
            duration=180.0,
            format_type="9:16",
            resolution="1080x1920",
            target_platforms=["tiktok", "instagram"],
            assembly_parameters={"effects": ["fade_in", "fade_out"]},
            generation_cost=0.50
        )
        
        test_session.add(video)
        await test_session.commit()
        await test_session.refresh(video)
        
        assert video.id is not None
        assert video.story_id == sample_story.id
        assert video.title == "My Test Video"
        assert video.duration == 180.0
        assert video.format_type == "9:16"
        assert video.target_platforms == ["tiktok", "instagram"]
        assert video.assembly_parameters["effects"] == ["fade_in", "fade_out"]


class TestPlatformUploadModel:
    """Tests for PlatformUpload model"""
    
    async def test_create_platform_upload(self, test_session, sample_video):
        """Test creating a platform upload"""
        upload = PlatformUpload(
            video_id=sample_video.id,
            platform=PlatformType.YOUTUBE,
            platform_video_id="abc123",
            upload_url="https://youtube.com/watch?v=abc123",
            upload_title="My YouTube Video",
            upload_description="Description for YouTube",
            hashtags=["#ai", "#story", "#video"],
            upload_metadata={"category": "Entertainment"}
        )
        
        test_session.add(upload)
        await test_session.commit()
        await test_session.refresh(upload)
        
        assert upload.id is not None
        assert upload.video_id == sample_video.id
        assert upload.platform == PlatformType.YOUTUBE
        assert upload.platform_video_id == "abc123"
        assert upload.hashtags == ["#ai", "#story", "#video"]
    
    async def test_unique_video_platform_constraint(self, test_session, sample_video):
        """Test unique constraint on video_id + platform"""
        # Create first upload
        upload1 = PlatformUpload(
            video_id=sample_video.id,
            platform=PlatformType.YOUTUBE,
            platform_video_id="abc123"
        )
        test_session.add(upload1)
        await test_session.commit()
        
        # Try to create duplicate upload for same video + platform
        upload2 = PlatformUpload(
            video_id=sample_video.id,
            platform=PlatformType.YOUTUBE,
            platform_video_id="def456"
        )
        test_session.add(upload2)
        
        with pytest.raises(Exception):  # Should raise integrity error
            await test_session.commit()


class TestAnalyticsDataModel:
    """Tests for AnalyticsData model"""
    
    async def test_create_analytics_data(self, test_session, sample_video):
        """Test creating analytics data"""
        # First create a platform upload
        upload = PlatformUpload(
            video_id=sample_video.id,
            platform=PlatformType.YOUTUBE,
            platform_video_id="abc123"
        )
        test_session.add(upload)
        await test_session.commit()
        await test_session.refresh(upload)
        
        # Create analytics data
        analytics = AnalyticsData(
            platform_upload_id=upload.id,
            views=1000,
            likes=50,
            comments=10,
            shares=5,
            completion_rate=75.5,
            ad_revenue=2.50,
            creator_fund_revenue=1.25,
            ctr=3.2,
            engagement_rate=6.5,
            data_collected_at=datetime.utcnow(),
            raw_analytics={"detailed_metrics": "data"}
        )
        
        test_session.add(analytics)
        await test_session.commit()
        await test_session.refresh(analytics)
        
        assert analytics.id is not None
        assert analytics.platform_upload_id == upload.id
        assert analytics.views == 1000
        assert analytics.total_revenue == 3.75  # ad_revenue + creator_fund_revenue
        assert analytics.completion_rate == 75.5
        assert analytics.raw_analytics["detailed_metrics"] == "data"


class TestModelConfigurationModel:
    """Tests for ModelConfiguration model"""
    
    async def test_create_model_configuration(self, test_session):
        """Test creating a model configuration"""
        config = ModelConfiguration(
            name="GPT-4 Story Config",
            description="Configuration for GPT-4 story generation",
            config_type="story",
            provider="openai",
            model_name="gpt-4",
            parameters={
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9
            },
            cost_per_unit=0.03,
            performance_metrics={"quality_score": 8.5, "speed": "fast"},
            version="1.0",
            is_active=True,
            is_default=True,
            traffic_percentage=100.0
        )
        
        test_session.add(config)
        await test_session.commit()
        await test_session.refresh(config)
        
        assert config.id is not None
        assert config.name == "GPT-4 Story Config"
        assert config.config_type == "story"
        assert config.parameters["temperature"] == 0.7
        assert config.is_active is True
        assert config.is_default is True
    
    async def test_unique_name_version_constraint(self, test_session):
        """Test unique constraint on name + version"""
        # Create first config
        config1 = ModelConfiguration(
            name="Test Config",
            config_type="story",
            parameters={"param": "value1"},
            version="1.0"
        )
        test_session.add(config1)
        await test_session.commit()
        
        # Try to create duplicate name + version
        config2 = ModelConfiguration(
            name="Test Config",
            config_type="story",
            parameters={"param": "value2"},
            version="1.0"
        )
        test_session.add(config2)
        
        with pytest.raises(Exception):  # Should raise integrity error
            await test_session.commit()


class TestEnumValues:
    """Tests for enum values"""
    
    def test_content_status_enum(self):
        """Test ContentStatus enum values"""
        assert ContentStatus.PENDING.value == "pending"
        assert ContentStatus.PROCESSING.value == "processing"
        assert ContentStatus.COMPLETED.value == "completed"
        assert ContentStatus.FAILED.value == "failed"
        assert ContentStatus.MODERATION_PENDING.value == "moderation_pending"
        assert ContentStatus.MODERATION_FAILED.value == "moderation_failed"
        assert ContentStatus.READY_FOR_DISTRIBUTION.value == "ready_for_distribution"
        assert ContentStatus.DISTRIBUTED.value == "distributed"
    
    def test_platform_type_enum(self):
        """Test PlatformType enum values"""
        assert PlatformType.YOUTUBE.value == "youtube"
        assert PlatformType.INSTAGRAM.value == "instagram"
        assert PlatformType.TIKTOK.value == "tiktok"
        assert PlatformType.FACEBOOK.value == "facebook"
    
    def test_media_type_enum(self):
        """Test MediaType enum values"""
        assert MediaType.AUDIO.value == "audio"
        assert MediaType.IMAGE.value == "image"
        assert MediaType.VIDEO.value == "video"
        assert MediaType.MUSIC.value == "music"