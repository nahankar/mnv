"""Tests for Pydantic schemas"""
import pytest
from datetime import datetime
from pydantic import ValidationError
import uuid

from shared.schemas import (
    StoryGenerationRequest, StoryResponse, StoryUpdate,
    MediaAssetRequest, MediaAssetResponse,
    VideoGenerationRequest, VideoResponse,
    PlatformUploadRequest, PlatformUploadResponse,
    AnalyticsDataRequest, AnalyticsDataResponse,
    ModelConfigurationRequest, ModelConfigurationResponse,
    PaginationParams, StoryFilters, ErrorResponse, HealthCheckResponse
)
from shared.models import ContentStatus, PlatformType, MediaType


class TestStorySchemas:
    """Tests for story-related schemas"""
    
    def test_story_generation_request_valid(self):
        """Test valid story generation request"""
        request = StoryGenerationRequest(
            genre="fantasy",
            theme="adventure",
            target_length=500,
            llm_provider="openai",
            generation_parameters={"temperature": 0.7}
        )
        
        assert request.genre == "fantasy"
        assert request.theme == "adventure"
        assert request.target_length == 500
        assert request.llm_provider == "openai"
        assert request.generation_parameters["temperature"] == 0.7
    
    def test_story_generation_request_optional_fields(self):
        """Test story generation request with optional fields"""
        request = StoryGenerationRequest()
        
        assert request.genre is None
        assert request.theme is None
        assert request.target_length is None
        assert request.llm_provider is None
        assert request.generation_parameters is None
    
    def test_story_generation_request_validation(self):
        """Test story generation request validation"""
        # Test target_length validation
        with pytest.raises(ValidationError):
            StoryGenerationRequest(target_length=10)  # Too small
        
        with pytest.raises(ValidationError):
            StoryGenerationRequest(target_length=5000)  # Too large
    
    def test_story_response_valid(self):
        """Test valid story response"""
        story_id = uuid.uuid4()
        now = datetime.utcnow()
        
        response = StoryResponse(
            id=story_id,
            title="Test Story",
            content="Story content",
            genre="sci-fi",
            theme="space",
            target_length=100,
            actual_length=95,
            llm_provider="openai",
            model_name="gpt-4",
            generation_parameters={"temperature": 0.8},
            generation_cost=0.05,
            status=ContentStatus.COMPLETED,
            created_at=now,
            updated_at=now
        )
        
        assert response.id == story_id
        assert response.title == "Test Story"
        assert response.status == ContentStatus.COMPLETED
        assert response.generation_cost == 0.05
    
    def test_story_update_partial(self):
        """Test partial story update"""
        update = StoryUpdate(
            title="Updated Title",
            status=ContentStatus.PROCESSING
        )
        
        assert update.title == "Updated Title"
        assert update.status == ContentStatus.PROCESSING
        assert update.content is None
        assert update.genre is None


class TestMediaAssetSchemas:
    """Tests for media asset schemas"""
    
    def test_media_asset_request_valid(self):
        """Test valid media asset request"""
        story_id = uuid.uuid4()
        
        request = MediaAssetRequest(
            story_id=story_id,
            asset_type=MediaType.AUDIO,
            provider="elevenlabs",
            generation_parameters={"voice": "adam"},
            prompt_override="Custom prompt"
        )
        
        assert request.story_id == story_id
        assert request.asset_type == MediaType.AUDIO
        assert request.provider == "elevenlabs"
        assert request.generation_parameters["voice"] == "adam"
        assert request.prompt_override == "Custom prompt"
    
    def test_media_asset_response_valid(self):
        """Test valid media asset response"""
        asset_id = uuid.uuid4()
        story_id = uuid.uuid4()
        now = datetime.utcnow()
        
        response = MediaAssetResponse(
            id=asset_id,
            story_id=story_id,
            asset_type=MediaType.IMAGE,
            file_path="/path/to/image.jpg",
            file_size=1024000,
            duration=None,
            metadata={"width": 1920, "height": 1080},
            provider="dall-e",
            model_name="dall-e-3",
            generation_parameters={"style": "vivid"},
            generation_cost=0.08,
            prompt_used="Generate an image",
            status=ContentStatus.COMPLETED,
            created_at=now,
            updated_at=now
        )
        
        assert response.id == asset_id
        assert response.story_id == story_id
        assert response.asset_type == MediaType.IMAGE
        assert response.file_size == 1024000
        assert response.metadata["width"] == 1920


class TestVideoSchemas:
    """Tests for video schemas"""
    
    def test_video_generation_request_valid(self):
        """Test valid video generation request"""
        story_id = uuid.uuid4()
        
        request = VideoGenerationRequest(
            story_id=story_id,
            title="Test Video",
            description="Video description",
            format_type="16:9",
            resolution="1920x1080",
            target_platforms=[PlatformType.YOUTUBE, PlatformType.INSTAGRAM],
            assembly_parameters={"transitions": True}
        )
        
        assert request.story_id == story_id
        assert request.title == "Test Video"
        assert request.format_type == "16:9"
        assert request.resolution == "1920x1080"
        assert len(request.target_platforms) == 2
        assert PlatformType.YOUTUBE in request.target_platforms
    
    def test_video_generation_request_format_validation(self):
        """Test video format validation"""
        story_id = uuid.uuid4()
        
        # Valid formats
        for format_type in ["16:9", "9:16", "1:1"]:
            request = VideoGenerationRequest(
                story_id=story_id,
                title="Test",
                format_type=format_type
            )
            assert request.format_type == format_type
        
        # Invalid format
        with pytest.raises(ValidationError):
            VideoGenerationRequest(
                story_id=story_id,
                title="Test",
                format_type="4:3"  # Not allowed
            )


class TestPlatformUploadSchemas:
    """Tests for platform upload schemas"""
    
    def test_platform_upload_request_valid(self):
        """Test valid platform upload request"""
        video_id = uuid.uuid4()
        
        request = PlatformUploadRequest(
            video_id=video_id,
            platform=PlatformType.TIKTOK,
            upload_title="TikTok Video",
            upload_description="Description for TikTok",
            hashtags=["#ai", "#story", "#video"],
            upload_metadata={"category": "Entertainment"}
        )
        
        assert request.video_id == video_id
        assert request.platform == PlatformType.TIKTOK
        assert request.upload_title == "TikTok Video"
        assert len(request.hashtags) == 3
        assert "#ai" in request.hashtags
    
    def test_platform_upload_response_valid(self):
        """Test valid platform upload response"""
        upload_id = uuid.uuid4()
        video_id = uuid.uuid4()
        now = datetime.utcnow()
        
        response = PlatformUploadResponse(
            id=upload_id,
            video_id=video_id,
            platform=PlatformType.FACEBOOK,
            platform_video_id="fb123456",
            upload_url="https://facebook.com/video/fb123456",
            upload_title="Facebook Video",
            upload_description="Description",
            hashtags=["#facebook"],
            upload_metadata={"category": "Entertainment"},
            status=ContentStatus.DISTRIBUTED,
            upload_started_at=now,
            upload_completed_at=now,
            created_at=now,
            updated_at=now,
            error_message=None,
            retry_count=0
        )
        
        assert response.id == upload_id
        assert response.platform == PlatformType.FACEBOOK
        assert response.platform_video_id == "fb123456"
        assert response.status == ContentStatus.DISTRIBUTED
        assert response.retry_count == 0


class TestAnalyticsSchemas:
    """Tests for analytics schemas"""
    
    def test_analytics_data_request_valid(self):
        """Test valid analytics data request"""
        upload_id = uuid.uuid4()
        
        request = AnalyticsDataRequest(
            platform_upload_id=upload_id,
            views=1000,
            likes=50,
            comments=10,
            shares=5,
            completion_rate=75.5,
            ad_revenue=2.50,
            creator_fund_revenue=1.25,
            ctr=3.2,
            engagement_rate=6.5,
            raw_analytics={"detailed": "data"}
        )
        
        assert request.platform_upload_id == upload_id
        assert request.views == 1000
        assert request.completion_rate == 75.5
        assert request.ad_revenue == 2.50
        assert request.raw_analytics["detailed"] == "data"
    
    def test_analytics_data_request_validation(self):
        """Test analytics data validation"""
        upload_id = uuid.uuid4()
        
        # Test negative values validation
        with pytest.raises(ValidationError):
            AnalyticsDataRequest(
                platform_upload_id=upload_id,
                views=-1  # Should be >= 0
            )
        
        # Test completion rate validation
        with pytest.raises(ValidationError):
            AnalyticsDataRequest(
                platform_upload_id=upload_id,
                completion_rate=150.0  # Should be <= 100
            )


class TestModelConfigurationSchemas:
    """Tests for model configuration schemas"""
    
    def test_model_configuration_request_valid(self):
        """Test valid model configuration request"""
        request = ModelConfigurationRequest(
            name="GPT-4 Config",
            description="Configuration for GPT-4",
            config_type="story",
            provider="openai",
            model_name="gpt-4",
            parameters={"temperature": 0.7, "max_tokens": 1000},
            cost_per_unit=0.03,
            performance_metrics={"quality": 8.5},
            version="1.0",
            is_active=True,
            is_default=False,
            ab_test_group="group_a",
            traffic_percentage=50.0
        )
        
        assert request.name == "GPT-4 Config"
        assert request.config_type == "story"
        assert request.parameters["temperature"] == 0.7
        assert request.traffic_percentage == 50.0
    
    def test_model_configuration_request_validation(self):
        """Test model configuration validation"""
        # Test invalid config_type
        with pytest.raises(ValidationError):
            ModelConfigurationRequest(
                name="Test",
                config_type="invalid_type",  # Not in allowed values
                parameters={"param": "value"},
                version="1.0"
            )
        
        # Test traffic percentage validation
        with pytest.raises(ValidationError):
            ModelConfigurationRequest(
                name="Test",
                config_type="story",
                parameters={"param": "value"},
                version="1.0",
                traffic_percentage=150.0  # Should be <= 100
            )


class TestUtilitySchemas:
    """Tests for utility schemas"""
    
    def test_pagination_params_valid(self):
        """Test valid pagination parameters"""
        params = PaginationParams(page=2, size=50)
        
        assert params.page == 2
        assert params.size == 50
    
    def test_pagination_params_validation(self):
        """Test pagination parameters validation"""
        # Test invalid page
        with pytest.raises(ValidationError):
            PaginationParams(page=0)  # Should be >= 1
        
        # Test invalid size
        with pytest.raises(ValidationError):
            PaginationParams(size=0)  # Should be >= 1
        
        with pytest.raises(ValidationError):
            PaginationParams(size=200)  # Should be <= 100
    
    def test_story_filters_valid(self):
        """Test valid story filters"""
        now = datetime.utcnow()
        
        filters = StoryFilters(
            status=ContentStatus.COMPLETED,
            genre="fantasy",
            theme="adventure",
            llm_provider="openai",
            created_after=now,
            created_before=now
        )
        
        assert filters.status == ContentStatus.COMPLETED
        assert filters.genre == "fantasy"
        assert filters.llm_provider == "openai"
    
    def test_error_response_valid(self):
        """Test valid error response"""
        error = ErrorResponse(
            error="ValidationError",
            message="Invalid input data",
            details={"field": "value"},
            correlation_id="123-456-789"
        )
        
        assert error.error == "ValidationError"
        assert error.message == "Invalid input data"
        assert error.details["field"] == "value"
        assert error.correlation_id == "123-456-789"
    
    def test_health_check_response_valid(self):
        """Test valid health check response"""
        now = datetime.utcnow()
        
        health = HealthCheckResponse(
            status="healthy",
            service="test-service",
            timestamp=now,
            version="1.0.0",
            dependencies={"database": "healthy", "redis": "healthy"}
        )
        
        assert health.status == "healthy"
        assert health.service == "test-service"
        assert health.version == "1.0.0"
        assert health.dependencies["database"] == "healthy"
    
    def test_health_check_response_validation(self):
        """Test health check response validation"""
        now = datetime.utcnow()
        
        # Test invalid status
        with pytest.raises(ValidationError):
            HealthCheckResponse(
                status="invalid",  # Should be "healthy" or "unhealthy"
                service="test-service",
                timestamp=now
            )