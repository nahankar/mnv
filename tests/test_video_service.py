"""
Unit tests for Video Assembly Service
"""

import pytest
import httpx
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import os

# Import the video service components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'video-service'))

from main import (
    VideoProcessor, VideoService, VideoFormat, VideoQuality, VideoEffect, Platform,
    VideoAsset, VideoAssemblyRequest, VideoAssemblyResult, PlatformMetadata
)


class TestVideoProcessor:
    """Test video processor functionality"""
    
    @pytest.fixture
    def video_processor(self):
        return VideoProcessor()
    
    @pytest.fixture
    def sample_assets(self):
        """Create sample video assets for testing"""
        return [
            VideoAsset(
                asset_id="img-1",
                asset_type="image",
                file_path="/tmp/test_image.jpg",
                duration=3.0
            ),
            VideoAsset(
                asset_id="audio-1",
                asset_type="audio",
                file_path="/tmp/test_audio.wav",
                volume=1.0
            ),
            VideoAsset(
                asset_id="music-1",
                asset_type="music",
                file_path="/tmp/test_music.mp3",
                volume=0.3
            )
        ]
    
    @pytest.fixture
    def sample_request(self, sample_assets):
        """Create sample video assembly request"""
        return VideoAssemblyRequest(
            story_id="story-123",
            title="Test Video",
            description="A test video for unit testing",
            format=VideoFormat.LANDSCAPE_16_9,
            quality=VideoQuality.HIGH,
            assets=sample_assets,
            platform=Platform.YOUTUBE
        )
    
    def test_video_processor_init(self, video_processor):
        """Test video processor initialization"""
        assert video_processor is not None
        assert len(video_processor.supported_formats) == 4
        assert len(video_processor.quality_settings) == 4
    
    def test_supported_formats(self, video_processor):
        """Test supported video formats"""
        formats = video_processor.supported_formats
        
        # Test landscape format
        assert VideoFormat.LANDSCAPE_16_9 in formats
        assert formats[VideoFormat.LANDSCAPE_16_9]["width"] == 1920
        assert formats[VideoFormat.LANDSCAPE_16_9]["height"] == 1080
        
        # Test portrait format
        assert VideoFormat.PORTRAIT_9_16 in formats
        assert formats[VideoFormat.PORTRAIT_9_16]["width"] == 1080
        assert formats[VideoFormat.PORTRAIT_9_16]["height"] == 1920
        
        # Test square format
        assert VideoFormat.SQUARE_1_1 in formats
        assert formats[VideoFormat.SQUARE_1_1]["width"] == 1080
        assert formats[VideoFormat.SQUARE_1_1]["height"] == 1080
    
    def test_quality_settings(self, video_processor):
        """Test quality settings"""
        settings = video_processor.quality_settings
        
        # Test low quality
        assert VideoQuality.LOW in settings
        assert settings[VideoQuality.LOW]["bitrate"] == "1000k"
        assert settings[VideoQuality.LOW]["crf"] == 28
        
        # Test high quality
        assert VideoQuality.HIGH in settings
        assert settings[VideoQuality.HIGH]["bitrate"] == "4000k"
        assert settings[VideoQuality.HIGH]["crf"] == 18
    
    @pytest.mark.asyncio
    async def test_process_image_asset(self, video_processor):
        """Test image asset processing"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'fake image data')
            temp_path = f.name
        
        try:
            asset = VideoAsset(
                asset_id="test-img",
                asset_type="image",
                file_path=temp_path,
                duration=5.0
            )
            
            # Mock ImageClip to avoid actual image processing
            with patch('main.ImageClip') as mock_image_clip:
                mock_clip = Mock()
                mock_image_clip.return_value = mock_clip
                mock_clip.set_duration.return_value = mock_clip
                mock_clip.set_start.return_value = mock_clip
                
                result = await video_processor._process_image_asset(asset, Path("/tmp"))
                
                assert result["type"] == "image"
                assert result["asset"] == asset
                assert result["clip"] == mock_clip
                
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_process_audio_asset(self, video_processor):
        """Test audio asset processing"""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'fake audio data')
            temp_path = f.name
        
        try:
            asset = VideoAsset(
                asset_id="test-audio",
                asset_type="audio",
                file_path=temp_path,
                volume=0.8
            )
            
            # Mock AudioFileClip to avoid actual audio processing
            with patch('main.AudioFileClip') as mock_audio_clip:
                mock_clip = Mock()
                mock_audio_clip.return_value = mock_clip
                mock_clip.volumex.return_value = mock_clip
                mock_clip.set_start.return_value = mock_clip
                
                result = await video_processor._process_audio_asset(asset, Path("/tmp"))
                
                assert result["type"] == "audio"
                assert result["asset"] == asset
                assert result["clip"] == mock_clip
                
        finally:
            os.unlink(temp_path)
    
    def test_get_output_format(self, video_processor):
        """Test output format determination"""
        assert video_processor._get_output_format(Platform.YOUTUBE) == "mp4"
        assert video_processor._get_output_format(Platform.INSTAGRAM) == "mp4"
        assert video_processor._get_output_format(Platform.TIKTOK) == "mp4"
        assert video_processor._get_output_format(Platform.FACEBOOK) == "mp4"
    
    @pytest.mark.asyncio
    async def test_generate_youtube_tags(self, video_processor):
        """Test YouTube tag generation"""
        title = "Amazing Story"
        metadata = {"genre": "adventure", "mood": "exciting"}
        
        tags = video_processor._generate_youtube_tags(title, metadata)
        
        assert "story" in tags
        assert "ai generated" in tags
        assert "creative" in tags
        assert "adventure" in tags
        assert "exciting" in tags
        assert "#shorts" in tags
        assert len(tags) <= 15  # YouTube limit
    
    @pytest.mark.asyncio
    async def test_generate_instagram_tags(self, video_processor):
        """Test Instagram tag generation"""
        title = "Creative Story"
        metadata = {}
        
        tags = video_processor._generate_instagram_tags(title, metadata)
        
        assert "story" in tags
        assert "creative" in tags
        assert "ai" in tags
        assert "#story" in tags
        assert "#creative" in tags
        assert len(tags) <= 30  # Instagram limit
    
    @pytest.mark.asyncio
    async def test_generate_tiktok_tags(self, video_processor):
        """Test TikTok tag generation"""
        title = "Viral Story"
        metadata = {}
        
        tags = video_processor._generate_tiktok_tags(title, metadata)
        
        assert "story" in tags
        assert "fyp" in tags
        assert "viral" in tags
        assert "#fyp" in tags
        assert "#foryou" in tags
        assert len(tags) <= 20  # TikTok limit


class TestVideoService:
    """Test video service functionality"""
    
    @pytest.fixture
    def mock_db_manager(self):
        return Mock()
    
    @pytest.fixture
    def video_service(self, mock_db_manager):
        return VideoService(mock_db_manager)
    
    @pytest.fixture
    def valid_request(self):
        """Create a valid video assembly request"""
        return VideoAssemblyRequest(
            story_id="story-123",
            title="Test Video",
            format=VideoFormat.LANDSCAPE_16_9,
            quality=VideoQuality.HIGH,
            assets=[
                VideoAsset(
                    asset_id="img-1",
                    asset_type="image",
                    file_path="/tmp/test_image.jpg",
                    duration=3.0
                )
            ],
            platform=Platform.YOUTUBE
        )
    
    @pytest.mark.asyncio
    async def test_validate_request_valid(self, video_service, valid_request):
        """Test request validation with valid request"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'fake image data')
            temp_path = f.name
        
        try:
            valid_request.assets[0].file_path = temp_path
            await video_service._validate_request(valid_request)
            # Should not raise any exception
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_validate_request_no_assets(self, video_service):
        """Test request validation with no assets"""
        request = VideoAssemblyRequest(
            story_id="story-123",
            title="Test Video",
            format=VideoFormat.LANDSCAPE_16_9,
            quality=VideoQuality.HIGH,
            assets=[],
            platform=Platform.YOUTUBE
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await video_service._validate_request(request)
        
        assert exc_info.value.status_code == 400
        assert "At least one asset is required" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_request_missing_file(self, video_service, valid_request):
        """Test request validation with missing file"""
        valid_request.assets[0].file_path = "/nonexistent/file.jpg"
        
        with pytest.raises(HTTPException) as exc_info:
            await video_service._validate_request(valid_request)
        
        assert exc_info.value.status_code == 400
        assert "Asset file not found" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_request_duration_limit_exceeded(self, video_service, valid_request):
        """Test request validation with excessive duration limit"""
        valid_request.duration_limit = 1000  # 10+ minutes
        
        with pytest.raises(HTTPException) as exc_info:
            await video_service._validate_request(valid_request)
        
        assert exc_info.value.status_code == 400
        assert "Duration limit cannot exceed 600 seconds" in str(exc_info.value.detail)


class TestVideoServiceEndpoints:
    """Test video service API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "video-service"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/")
            assert response.status_code == 200
            data = response.json()
            assert "Video Assembly Service API" in data["message"]
            assert data["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_formats_endpoint(self):
        """Test supported formats endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/formats")
            assert response.status_code == 200
            data = response.json()
            assert "formats" in data
            assert "quality_settings" in data
            assert "platforms" in data
            assert "effects" in data
            
            # Check specific formats
            formats = data["formats"]
            assert "16:9" in formats
            assert "9:16" in formats
            assert "1:1" in formats
    
    @pytest.mark.asyncio
    async def test_platform_metadata_youtube(self):
        """Test YouTube platform metadata template"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/platforms/youtube/metadata")
            assert response.status_code == 200
            data = response.json()
            assert "title_max_length" in data
            assert "description_max_length" in data
            assert "tags_max_count" in data
            assert "categories" in data
            assert data["title_max_length"] == 100
            assert data["tags_max_count"] == 15
    
    @pytest.mark.asyncio
    async def test_platform_metadata_instagram(self):
        """Test Instagram platform metadata template"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/platforms/instagram/metadata")
            assert response.status_code == 200
            data = response.json()
            assert "title_max_length" in data
            assert "description_max_length" in data
            assert "tags_max_count" in data
            assert "formats" in data
            assert data["title_max_length"] == 125
            assert data["tags_max_count"] == 30
    
    @pytest.mark.asyncio
    async def test_platform_metadata_tiktok(self):
        """Test TikTok platform metadata template"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/platforms/tiktok/metadata")
            assert response.status_code == 200
            data = response.json()
            assert "title_max_length" in data
            assert "description_max_length" in data
            assert "tags_max_count" in data
            assert "formats" in data
            assert data["title_max_length"] == 150
            assert data["tags_max_count"] == 20
            assert "9:16" in data["formats"]


class TestVideoAssembly:
    """Test video assembly functionality"""
    
    @pytest.mark.asyncio
    async def test_video_assembly_request_validation(self):
        """Test video assembly request validation"""
        # Test valid request
        valid_request = VideoAssemblyRequest(
            story_id="story-123",
            title="Test Video",
            format=VideoFormat.LANDSCAPE_16_9,
            quality=VideoQuality.HIGH,
            assets=[
                VideoAsset(
                    asset_id="img-1",
                    asset_type="image",
                    file_path="/tmp/test.jpg",
                    duration=3.0
                )
            ],
            platform=Platform.YOUTUBE
        )
        
        assert valid_request.story_id == "story-123"
        assert valid_request.title == "Test Video"
        assert valid_request.format == VideoFormat.LANDSCAPE_16_9
        assert valid_request.quality == VideoQuality.HIGH
        assert len(valid_request.assets) == 1
        assert valid_request.platform == Platform.YOUTUBE
    
    @pytest.mark.asyncio
    async def test_video_asset_validation(self):
        """Test video asset validation"""
        asset = VideoAsset(
            asset_id="test-asset",
            asset_type="image",
            file_path="/tmp/test.jpg",
            duration=5.0,
            volume=0.8,
            effects=[VideoEffect.FADE_IN]
        )
        
        assert asset.asset_id == "test-asset"
        assert asset.asset_type == "image"
        assert asset.file_path == "/tmp/test.jpg"
        assert asset.duration == 5.0
        assert asset.volume == 0.8
        assert VideoEffect.FADE_IN in asset.effects


class TestVideoEffects:
    """Test video effects functionality"""
    
    def test_video_effects_enum(self):
        """Test video effects enumeration"""
        effects = [
            VideoEffect.NONE,
            VideoEffect.FADE_IN,
            VideoEffect.FADE_OUT,
            VideoEffect.ZOOM,
            VideoEffect.PAN,
            VideoEffect.COLOR_CORRECTION,
            VideoEffect.BRIGHTNESS,
            VideoEffect.CONTRAST
        ]
        
        assert len(effects) == 8
        assert VideoEffect.FADE_IN.value == "fade_in"
        assert VideoEffect.ZOOM.value == "zoom"
    
    @pytest.mark.asyncio
    async def test_apply_effects_mock(self):
        """Test applying effects with mocked video clip"""
        processor = VideoProcessor()
        
        # Mock video clip
        mock_clip = Mock()
        mock_clip.fadein.return_value = mock_clip
        mock_clip.fadeout.return_value = mock_clip
        mock_clip.fx.return_value = mock_clip
        
        effects = [VideoEffect.FADE_IN, VideoEffect.FADE_OUT, VideoEffect.ZOOM]
        
        result = await processor._apply_effects(mock_clip, effects)
        
        assert result == mock_clip
        mock_clip.fadein.assert_called_once_with(1.0)
        mock_clip.fadeout.assert_called_once_with(1.0)
        mock_clip.fx.assert_called()


class TestPlatformMetadata:
    """Test platform metadata generation"""
    
    @pytest.mark.asyncio
    async def test_platform_metadata_creation(self):
        """Test platform metadata creation"""
        metadata = PlatformMetadata(
            title="Test Video",
            description="A test video description",
            tags=["tag1", "tag2", "tag3"],
            category="Entertainment",
            privacy="public"
        )
        
        assert metadata.title == "Test Video"
        assert metadata.description == "A test video description"
        assert len(metadata.tags) == 3
        assert metadata.category == "Entertainment"
        assert metadata.privacy == "public"
    
    @pytest.mark.asyncio
    async def test_platform_metadata_serialization(self):
        """Test platform metadata serialization"""
        metadata = PlatformMetadata(
            title="Test Video",
            description="A test video description",
            tags=["tag1", "tag2"],
            category="Entertainment"
        )
        
        data = metadata.dict()
        
        assert data["title"] == "Test Video"
        assert data["description"] == "A test video description"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["category"] == "Entertainment"
        assert data["privacy"] == "public"  # Default value


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_video_assembly_missing_assets(self):
        """Test video assembly with missing assets"""
        request = VideoAssemblyRequest(
            story_id="story-123",
            title="Test Video",
            format=VideoFormat.LANDSCAPE_16_9,
            quality=VideoQuality.HIGH,
            assets=[],
            platform=Platform.YOUTUBE
        )
        
        # This should be caught by validation
        assert len(request.assets) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_video_format(self):
        """Test invalid video format handling"""
        # Test with invalid format string
        with pytest.raises(ValueError):
            VideoFormat("invalid_format")
    
    @pytest.mark.asyncio
    async def test_invalid_quality_setting(self):
        """Test invalid quality setting handling"""
        # Test with invalid quality string
        with pytest.raises(ValueError):
            VideoQuality("invalid_quality")


class TestVideoFormats:
    """Test video format functionality"""
    
    def test_video_format_enum(self):
        """Test video format enumeration"""
        formats = [
            VideoFormat.LANDSCAPE_16_9,
            VideoFormat.PORTRAIT_9_16,
            VideoFormat.SQUARE_1_1,
            VideoFormat.STORY_9_16
        ]
        
        assert len(formats) == 4
        assert VideoFormat.LANDSCAPE_16_9.value == "16:9"
        assert VideoFormat.PORTRAIT_9_16.value == "9:16"
        assert VideoFormat.SQUARE_1_1.value == "1:1"
        assert VideoFormat.STORY_9_16.value == "9:16_story"
    
    def test_platform_enum(self):
        """Test platform enumeration"""
        platforms = [
            Platform.YOUTUBE,
            Platform.INSTAGRAM,
            Platform.TIKTOK,
            Platform.FACEBOOK,
            Platform.TWITTER
        ]
        
        assert len(platforms) == 5
        assert Platform.YOUTUBE.value == "youtube"
        assert Platform.INSTAGRAM.value == "instagram"
        assert Platform.TIKTOK.value == "tiktok"
        assert Platform.FACEBOOK.value == "facebook"
        assert Platform.TWITTER.value == "twitter"


if __name__ == "__main__":
    pytest.main([__file__])
