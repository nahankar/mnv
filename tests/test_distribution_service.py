"""
Unit tests for Platform Distribution Service
"""

import pytest
import httpx
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import os
from datetime import datetime

# Import the distribution service components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'distribution-service'))

from main import (
    DistributionProcessor, UploadStatus, UploadPriority, PlatformUploadRequest,
    BatchUploadRequest, UploadResult, UploadStatusResponse, PlatformLimits
)
from shared.models import PlatformType, ContentStatus


class TestDistributionProcessor:
    """Test distribution processor functionality"""
    
    @pytest.fixture
    def mock_db_manager(self):
        return Mock()
    
    @pytest.fixture
    def distribution_processor(self, mock_db_manager):
        return DistributionProcessor(mock_db_manager)
    
    @pytest.fixture
    def sample_video(self):
        """Create sample video for testing"""
        video = Mock()
        video.id = uuid.uuid4()
        video.file_path = "/tmp/test_video.mp4"
        video.status = ContentStatus.COMPLETED
        video.title = "Test Video"
        video.duration = 30.0
        return video
    
    @pytest.fixture
    def sample_upload_request(self):
        """Create sample upload request"""
        return PlatformUploadRequest(
            video_id=str(uuid.uuid4()),
            platform=PlatformType.YOUTUBE,
            title="Test Video Upload",
            description="A test video for platform distribution",
            hashtags=["#test", "#video", "#ai"],
            category="Entertainment",
            privacy="public",
            priority=UploadPriority.NORMAL,
            metadata={"test": True}
        )
    
    def test_distribution_processor_init(self, distribution_processor):
        """Test distribution processor initialization"""
        assert distribution_processor is not None
        assert len(distribution_processor.platform_clients) == 4
        assert PlatformType.YOUTUBE in distribution_processor.platform_clients
        assert PlatformType.INSTAGRAM in distribution_processor.platform_clients
        assert PlatformType.TIKTOK in distribution_processor.platform_clients
        assert PlatformType.FACEBOOK in distribution_processor.platform_clients
    
    @pytest.mark.asyncio
    async def test_upload_to_platform_success(self, distribution_processor, sample_upload_request, sample_video):
        """Test successful platform upload"""
        # Mock database operations
        distribution_processor._get_video_by_id = AsyncMock(return_value=sample_video)
        distribution_processor._check_platform_limits = AsyncMock()
        distribution_processor._create_upload_record = AsyncMock(return_value=Mock(id=uuid.uuid4(), created_at=datetime.utcnow()))
        distribution_processor._update_upload_record = AsyncMock()
        
        # Mock platform client
        mock_client = AsyncMock()
        mock_client.upload_video.return_value = {
            "video_id": "test_platform_id",
            "url": "https://platform.com/video/test_platform_id"
        }
        distribution_processor.platform_clients[PlatformType.YOUTUBE] = mock_client
        
        result = await distribution_processor.upload_to_platform(sample_upload_request)
        
        assert result.status == UploadStatus.COMPLETED
        assert result.platform_video_id == "test_platform_id"
        assert result.upload_url == "https://platform.com/video/test_platform_id"
        assert result.processing_time is not None
    
    @pytest.mark.asyncio
    async def test_upload_to_platform_video_not_found(self, distribution_processor, sample_upload_request):
        """Test upload with non-existent video"""
        distribution_processor._get_video_by_id = AsyncMock(return_value=None)
        
        result = await distribution_processor.upload_to_platform(sample_upload_request)
        
        assert result.status == UploadStatus.FAILED
        assert "not found" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_upload_to_platform_video_not_ready(self, distribution_processor, sample_upload_request, sample_video):
        """Test upload with video not ready for distribution"""
        sample_video.status = ContentStatus.PROCESSING
        distribution_processor._get_video_by_id = AsyncMock(return_value=sample_video)
        
        result = await distribution_processor.upload_to_platform(sample_upload_request)
        
        assert result.status == UploadStatus.FAILED
        assert "not ready" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_upload_to_multiple_platforms(self, distribution_processor, sample_video):
        """Test batch upload to multiple platforms"""
        request = BatchUploadRequest(
            video_id=str(sample_video.id),
            platforms=[PlatformType.YOUTUBE, PlatformType.INSTAGRAM],
            title="Batch Upload Test",
            description="Testing batch upload functionality",
            hashtags=["#test", "#batch"],
            priority=UploadPriority.HIGH
        )
        
        # Mock single platform upload
        mock_result = UploadResult(
            upload_id=str(uuid.uuid4()),
            video_id=str(sample_video.id),
            platform=PlatformType.YOUTUBE,
            status=UploadStatus.COMPLETED,
            created_at=datetime.utcnow()
        )
        
        distribution_processor.upload_to_platform = AsyncMock(return_value=mock_result)
        
        results = await distribution_processor.upload_to_multiple_platforms(request)
        
        assert len(results) == 2
        assert all(isinstance(result, UploadResult) for result in results)
        assert distribution_processor.upload_to_platform.call_count == 2
    
    @pytest.mark.asyncio
    async def test_upload_with_retry_success_on_second_attempt(self, distribution_processor, sample_upload_request, sample_video):
        """Test upload retry logic with success on second attempt"""
        # Mock client that fails first then succeeds
        mock_client = AsyncMock()
        mock_client.upload_video.side_effect = [
            Exception("Network error"),  # First attempt fails
            {"video_id": "success_id", "url": "https://platform.com/success"}  # Second attempt succeeds
        ]
        
        distribution_processor._update_upload_status = AsyncMock()
        
        # Mock sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await distribution_processor._upload_with_retry(
                mock_client, sample_video, sample_upload_request, Mock()
            )
        
        assert result["video_id"] == "success_id"
        assert mock_client.upload_video.call_count == 2
    
    @pytest.mark.asyncio
    async def test_upload_with_retry_max_attempts_exceeded(self, distribution_processor, sample_upload_request, sample_video):
        """Test upload retry logic when max attempts exceeded"""
        # Mock client that always fails
        mock_client = AsyncMock()
        mock_client.upload_video.side_effect = Exception("Persistent error")
        
        distribution_processor._update_upload_status = AsyncMock()
        
        # Mock sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Persistent error"):
                await distribution_processor._upload_with_retry(
                    mock_client, sample_video, sample_upload_request, Mock()
                )
        
        # Should attempt MAX_RETRY_ATTEMPTS times
        assert mock_client.upload_video.call_count == 3  # DEFAULT MAX_RETRY_ATTEMPTS
    
    @pytest.mark.asyncio
    async def test_get_upload_status_found(self, distribution_processor):
        """Test getting upload status for existing upload"""
        mock_upload = Mock()
        mock_upload.id = uuid.uuid4()
        mock_upload.video_id = uuid.uuid4()
        mock_upload.platform = PlatformType.YOUTUBE
        mock_upload.status = ContentStatus.COMPLETED
        mock_upload.platform_video_id = "test_id"
        mock_upload.upload_url = "https://youtube.com/watch?v=test_id"
        mock_upload.error_message = None
        mock_upload.retry_count = 0
        mock_upload.created_at = datetime.utcnow()
        mock_upload.updated_at = datetime.utcnow()
        
        # Mock database session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_upload
        mock_session.execute.return_value = mock_result
        
        distribution_processor.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        status = await distribution_processor.get_upload_status(str(mock_upload.id))
        
        assert status is not None
        assert status.status == UploadStatus.COMPLETED
        assert status.platform_video_id == "test_id"
        assert status.upload_url == "https://youtube.com/watch?v=test_id"
    
    @pytest.mark.asyncio
    async def test_get_upload_status_not_found(self, distribution_processor):
        """Test getting upload status for non-existent upload"""
        # Mock database session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        distribution_processor.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        status = await distribution_processor.get_upload_status(str(uuid.uuid4()))
        
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_platform_limits(self, distribution_processor):
        """Test getting platform limits"""
        youtube_limits = await distribution_processor.get_platform_limits(PlatformType.YOUTUBE)
        
        assert isinstance(youtube_limits, PlatformLimits)
        assert youtube_limits.daily_uploads == 100
        assert youtube_limits.file_size_mb == 256000
        assert youtube_limits.title_max_length == 100
        assert youtube_limits.hashtags_max_count == 15
        
        # Test Instagram limits
        instagram_limits = await distribution_processor.get_platform_limits(PlatformType.INSTAGRAM)
        assert instagram_limits.daily_uploads == 50
        assert instagram_limits.file_size_mb == 4000
        assert instagram_limits.hashtags_max_count == 30
        
        # Test TikTok limits
        tiktok_limits = await distribution_processor.get_platform_limits(PlatformType.TIKTOK)
        assert tiktok_limits.daily_uploads == 30
        assert tiktok_limits.duration_seconds == 600
        assert tiktok_limits.hashtags_max_count == 20
        
        # Test Facebook limits
        facebook_limits = await distribution_processor.get_platform_limits(PlatformType.FACEBOOK)
        assert facebook_limits.daily_uploads == 100
        assert facebook_limits.file_size_mb == 10000
        assert facebook_limits.hashtags_max_count == 10


class TestDistributionServiceEndpoints:
    """Test distribution service API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8007/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "distribution-service"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8007/")
            assert response.status_code == 200
            data = response.json()
            assert "Platform Distribution Service API" in data["message"]
            assert data["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_platform_limits_endpoint(self):
        """Test platform limits endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8007/platforms/youtube/limits")
            assert response.status_code == 200
            data = response.json()
            assert "daily_uploads" in data
            assert "file_size_mb" in data
            assert "title_max_length" in data
            assert data["daily_uploads"] == 100
            assert data["title_max_length"] == 100
    
    @pytest.mark.asyncio
    async def test_platform_formats_endpoint(self):
        """Test platform formats endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8007/platforms/youtube/formats")
            assert response.status_code == 200
            data = response.json()
            assert "video_formats" in data
            assert "audio_formats" in data
            assert "aspect_ratios" in data
            assert "mp4" in data["video_formats"]
            assert "16:9" in data["aspect_ratios"]


class TestPlatformClients:
    """Test platform client functionality"""
    
    @pytest.mark.asyncio
    async def test_youtube_client_init(self):
        """Test YouTube client initialization"""
        # Import YouTube client
        from platform_clients.youtube_client import YouTubeClient
        
        client = YouTubeClient()
        assert client is not None
        assert client.SCOPES == ['https://www.googleapis.com/auth/youtube.upload']
        assert client.API_SERVICE_NAME == 'youtube'
        assert client.API_VERSION == 'v3'
    
    @pytest.mark.asyncio
    async def test_youtube_client_upload_without_credentials(self):
        """Test YouTube upload without proper credentials"""
        from platform_clients.youtube_client import YouTubeClient
        
        client = YouTubeClient()
        
        with pytest.raises(Exception, match="not properly initialized"):
            await client.upload_video(
                video_path="/tmp/test.mp4",
                title="Test Video",
                description="Test description"
            )
    
    @pytest.mark.asyncio
    async def test_youtube_client_format_description(self):
        """Test YouTube description formatting"""
        from platform_clients.youtube_client import YouTubeClient
        
        client = YouTubeClient()
        
        # Test basic description
        desc = client._format_description("Basic description", [])
        assert desc == "Basic description"
        
        # Test description with hashtags
        desc = client._format_description("Basic description", ["test", "video", "ai"])
        assert "Basic description" in desc
        assert "#test" in desc
        assert "#video" in desc
        assert "#ai" in desc
        
        # Test hashtag formatting (removes # prefix if present)
        desc = client._format_description("Basic description", ["#test", "video"])
        assert "#test" in desc
        assert "#video" in desc
        # Should not double the # for #test
        assert "##test" not in desc


class TestUploadModels:
    """Test upload request/response models"""
    
    def test_platform_upload_request_validation(self):
        """Test platform upload request validation"""
        # Test valid request
        request = PlatformUploadRequest(
            video_id=str(uuid.uuid4()),
            platform=PlatformType.YOUTUBE,
            title="Test Video",
            description="Test description",
            hashtags=["#test", "#video"],
            privacy="public",
            priority=UploadPriority.NORMAL
        )
        
        assert request.video_id is not None
        assert request.platform == PlatformType.YOUTUBE
        assert request.title == "Test Video"
        assert request.privacy == "public"
        assert request.priority == UploadPriority.NORMAL
    
    def test_batch_upload_request_validation(self):
        """Test batch upload request validation"""
        request = BatchUploadRequest(
            video_id=str(uuid.uuid4()),
            platforms=[PlatformType.YOUTUBE, PlatformType.INSTAGRAM],
            title="Batch Test Video",
            description="Batch test description",
            hashtags=["#batch", "#test"],
            priority=UploadPriority.HIGH,
            platform_specific={
                "youtube": {"category": "Entertainment"},
                "instagram": {"aspect_ratio": "1:1"}
            }
        )
        
        assert len(request.platforms) == 2
        assert PlatformType.YOUTUBE in request.platforms
        assert PlatformType.INSTAGRAM in request.platforms
        assert request.priority == UploadPriority.HIGH
        assert "youtube" in request.platform_specific
        assert "instagram" in request.platform_specific
    
    def test_upload_result_creation(self):
        """Test upload result creation"""
        result = UploadResult(
            upload_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            platform=PlatformType.YOUTUBE,
            status=UploadStatus.COMPLETED,
            platform_video_id="test_video_id",
            upload_url="https://youtube.com/watch?v=test_video_id",
            processing_time=15.5,
            created_at=datetime.utcnow()
        )
        
        assert result.status == UploadStatus.COMPLETED
        assert result.platform_video_id == "test_video_id"
        assert result.processing_time == 15.5
        assert result.retry_count == 0  # Default value
    
    def test_upload_status_response_creation(self):
        """Test upload status response creation"""
        now = datetime.utcnow()
        
        response = UploadStatusResponse(
            upload_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            platform=PlatformType.INSTAGRAM,
            status=UploadStatus.UPLOADING,
            progress=0.65,
            platform_video_id="insta_id",
            upload_url="https://instagram.com/p/insta_id",
            retry_count=1,
            created_at=now,
            updated_at=now
        )
        
        assert response.status == UploadStatus.UPLOADING
        assert response.progress == 0.65
        assert response.platform == PlatformType.INSTAGRAM
        assert response.retry_count == 1
    
    def test_platform_limits_creation(self):
        """Test platform limits model"""
        limits = PlatformLimits(
            daily_uploads=100,
            file_size_mb=1000,
            duration_seconds=3600,
            title_max_length=100,
            description_max_length=5000,
            hashtags_max_count=15,
            current_usage={"daily_uploads": 5, "file_size_used": 2500}
        )
        
        assert limits.daily_uploads == 100
        assert limits.file_size_mb == 1000
        assert limits.current_usage["daily_uploads"] == 5
        assert limits.current_usage["file_size_used"] == 2500


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_unsupported_platform_error(self, distribution_processor, sample_upload_request, sample_video):
        """Test handling of unsupported platform"""
        # Remove client for testing
        sample_upload_request.platform = PlatformType.YOUTUBE
        distribution_processor.platform_clients.pop(PlatformType.YOUTUBE, None)
        
        distribution_processor._get_video_by_id = AsyncMock(return_value=sample_video)
        distribution_processor._check_platform_limits = AsyncMock()
        distribution_processor._create_upload_record = AsyncMock(return_value=Mock(id=uuid.uuid4(), created_at=datetime.utcnow()))
        
        result = await distribution_processor.upload_to_platform(sample_upload_request)
        
        assert result.status == UploadStatus.FAILED
        assert "not supported" in result.error_message.lower()
    
    def test_upload_status_enum_values(self):
        """Test upload status enum values"""
        assert UploadStatus.PENDING == "pending"
        assert UploadStatus.UPLOADING == "uploading"
        assert UploadStatus.PROCESSING == "processing"
        assert UploadStatus.COMPLETED == "completed"
        assert UploadStatus.FAILED == "failed"
        assert UploadStatus.CANCELLED == "cancelled"
    
    def test_upload_priority_enum_values(self):
        """Test upload priority enum values"""
        assert UploadPriority.LOW == "low"
        assert UploadPriority.NORMAL == "normal"
        assert UploadPriority.HIGH == "high"
        assert UploadPriority.URGENT == "urgent"
    
    @pytest.mark.asyncio
    async def test_missing_video_file_error(self):
        """Test handling of missing video file"""
        from platform_clients.youtube_client import YouTubeClient
        
        client = YouTubeClient()
        # Mock credentials to bypass initialization check
        client.youtube_service = Mock()
        
        with pytest.raises(Exception, match="not found"):
            await client.upload_video(
                video_path="/nonexistent/file.mp4",
                title="Test",
                description="Test"
            )


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_upload_workflow_simulation(self, distribution_processor, sample_video):
        """Test complete upload workflow simulation"""
        # Setup mocks for full workflow
        upload_request = PlatformUploadRequest(
            video_id=str(sample_video.id),
            platform=PlatformType.YOUTUBE,
            title="Integration Test Video",
            description="Testing full upload workflow",
            hashtags=["#integration", "#test"],
            category="Education",
            privacy="unlisted",
            priority=UploadPriority.NORMAL
        )
        
        # Mock all dependencies
        distribution_processor._get_video_by_id = AsyncMock(return_value=sample_video)
        distribution_processor._check_platform_limits = AsyncMock()
        
        upload_record = Mock()
        upload_record.id = uuid.uuid4()
        upload_record.created_at = datetime.utcnow()
        distribution_processor._create_upload_record = AsyncMock(return_value=upload_record)
        distribution_processor._update_upload_record = AsyncMock()
        
        # Mock platform client
        mock_client = AsyncMock()
        mock_client.upload_video.return_value = {
            "video_id": "integration_test_id",
            "url": "https://youtube.com/watch?v=integration_test_id",
            "title": "Integration Test Video",
            "status": "public"
        }
        distribution_processor.platform_clients[PlatformType.YOUTUBE] = mock_client
        
        # Execute upload
        result = await distribution_processor.upload_to_platform(upload_request)
        
        # Verify result
        assert result.status == UploadStatus.COMPLETED
        assert result.platform_video_id == "integration_test_id"
        assert result.upload_url == "https://youtube.com/watch?v=integration_test_id"
        assert result.processing_time is not None
        assert result.processing_time > 0
        
        # Verify all steps were called
        distribution_processor._get_video_by_id.assert_called_once()
        distribution_processor._check_platform_limits.assert_called_once()
        distribution_processor._create_upload_record.assert_called_once()
        distribution_processor._update_upload_record.assert_called_once()
        mock_client.upload_video.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_uploads_simulation(self, distribution_processor, sample_video):
        """Test concurrent uploads to different platforms"""
        # Create multiple upload requests
        requests = [
            PlatformUploadRequest(
                video_id=str(sample_video.id),
                platform=platform,
                title=f"Concurrent Test - {platform.value}",
                description=f"Testing concurrent upload to {platform.value}",
                hashtags=[f"#{platform.value}", "#concurrent"],
                priority=UploadPriority.HIGH
            )
            for platform in [PlatformType.YOUTUBE, PlatformType.INSTAGRAM, PlatformType.TIKTOK]
        ]
        
        # Mock dependencies
        distribution_processor._get_video_by_id = AsyncMock(return_value=sample_video)
        distribution_processor._check_platform_limits = AsyncMock()
        distribution_processor._create_upload_record = AsyncMock(
            side_effect=lambda *args, **kwargs: Mock(id=uuid.uuid4(), created_at=datetime.utcnow())
        )
        distribution_processor._update_upload_record = AsyncMock()
        
        # Mock platform clients
        for platform in [PlatformType.YOUTUBE, PlatformType.INSTAGRAM, PlatformType.TIKTOK]:
            mock_client = AsyncMock()
            mock_client.upload_video.return_value = {
                "video_id": f"{platform.value}_concurrent_id",
                "url": f"https://{platform.value}.com/video/concurrent_id"
            }
            distribution_processor.platform_clients[platform] = mock_client
        
        # Execute concurrent uploads
        tasks = [
            distribution_processor.upload_to_platform(request)
            for request in requests
        ]
        results = await asyncio.gather(*tasks)
        
        # Verify all uploads completed successfully
        assert len(results) == 3
        assert all(result.status == UploadStatus.COMPLETED for result in results)
        assert all(result.platform_video_id is not None for result in results)
        assert all(result.upload_url is not None for result in results)
        
        # Verify unique platform IDs
        platform_ids = [result.platform_video_id for result in results]
        assert len(set(platform_ids)) == 3  # All unique


if __name__ == "__main__":
    pytest.main([__file__])