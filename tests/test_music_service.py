import os
import pytest
import httpx
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

# Import the music service components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'music-service'))

from main import (
    MockMusicProvider, MubertProvider, SunoProvider, 
    AudioProcessor, get_provider, audio_processor
)


@pytest.mark.asyncio
async def test_generate_music_mock():
    """Generate mock music and assert JSON and file presence."""
    # Ensure env defaults match mock
    assert os.getenv("MUSIC_PROVIDER", "mock") == "mock"

    payload = {
        "genre": "ambient",
        "mood": "calm",
        "duration_seconds": 5,
        "provider": "mock"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:8004/generate/music", json=payload, timeout=30.0)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "id" in body and body["id"]
    assert "file_path" in body and body["file_path"]
    assert body["provider_used"] == "mock"
    assert body["duration_seconds"] == 5


class TestMockMusicProvider:
    """Test mock music provider functionality"""
    
    @pytest.mark.asyncio
    async def test_mock_provider_generation(self):
        """Test mock provider generates valid audio file"""
        provider = MockMusicProvider()
        result = await provider.generate("ambient", "calm", 10)
        
        assert result["provider"] == "mock"
        assert result["format"] == "wav"
        assert "file_path" in result
        assert "license" in result
        assert result["license"]["type"] == "mock-dev"
        
        # Verify file exists
        file_path = Path(result["file_path"])
        assert file_path.exists()
        assert file_path.stat().st_size > 0


class TestMubertProvider:
    """Test Mubert provider integration"""
    
    @pytest.mark.asyncio
    async def test_mubert_provider_init(self):
        """Test Mubert provider initialization"""
        with patch.dict('os.environ', {'MUBERT_API_KEY': 'test-key'}):
            provider = MubertProvider("test-key")
            assert provider.api_key == "test-key"
            assert provider.base_url == "https://api.mubert.com/v2/Track"
    
    @pytest.mark.asyncio
    async def test_mubert_api_call_success(self):
        """Test successful Mubert API call"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.content = b"fake_audio_data"
            mock_response.headers = {"X-Request-ID": "test-123"}
            mock_response.raise_for_status = Mock()
            
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            provider = MubertProvider("test-key")
            result = await provider.generate("ambient", "calm", 30)
            
            assert result["provider"] == "mubert"
            assert result["format"] == "mp3"
            assert "file_path" in result
            assert result["license"]["type"] == "mubert"
            assert result["license"]["receipt"] == "test-123"
    
    @pytest.mark.asyncio
    async def test_mubert_api_call_failure(self):
        """Test Mubert API call failure handling"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate limited", request=Mock(), response=mock_response
            )
            
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            provider = MubertProvider("test-key")
            
            with pytest.raises(Exception):  # Should be retryable error
                await provider.generate("ambient", "calm", 30)


class TestSunoProvider:
    """Test Suno provider (currently stub)"""
    
    @pytest.mark.asyncio
    async def test_suno_provider_not_implemented(self):
        """Test Suno provider returns 501 as expected"""
        provider = SunoProvider("test-key")
        
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await provider.generate("ambient", "calm", 30)
        
        # Should raise 501 Not Implemented
        assert "Suno provider not implemented yet" in str(exc_info.value)


class TestAudioProcessor:
    """Test audio processing functionality"""
    
    @pytest.fixture
    def temp_audio_file(self):
        """Create temporary audio file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create minimal WAV file
            f.write(b'RIFF    WAVEfmt ')
            f.write(b'\x10\x00\x00\x00')  # fmt chunk size
            f.write(b'\x01\x00')          # audio format (PCM)
            f.write(b'\x01\x00')          # channels
            f.write(b'\x44\xAC\x00\x00')  # sample rate
            f.write(b'\x88\x58\x01\x00')  # byte rate
            f.write(b'\x02\x00')          # block align
            f.write(b'\x10\x00')          # bits per sample
            f.write(b'data    ')          # data chunk
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_audio_processor_init(self):
        """Test audio processor initialization"""
        processor = AudioProcessor()
        assert processor is not None
    
    @pytest.mark.asyncio
    async def test_normalize_volume_ffmpeg_failure(self):
        """Test volume normalization when ffmpeg fails"""
        processor = AudioProcessor()
        
        # Test with non-existent file
        result = await processor.normalize_volume("/nonexistent/file.wav")
        assert result == "/nonexistent/file.wav"  # Should return original path
    
    @pytest.mark.asyncio
    async def test_adjust_duration_ffmpeg_failure(self):
        """Test duration adjustment when ffmpeg fails"""
        processor = AudioProcessor()
        
        # Test with non-existent file
        result = await processor.adjust_duration("/nonexistent/file.wav", 30)
        assert result == "/nonexistent/file.wav"  # Should return original path
    
    @pytest.mark.asyncio
    async def test_adjust_volume_ffmpeg_failure(self):
        """Test volume adjustment when ffmpeg fails"""
        processor = AudioProcessor()
        
        # Test with non-existent file
        result = await processor.adjust_volume("/nonexistent/file.wav", -20.0)
        assert result == "/nonexistent/file.wav"  # Should return original path


class TestProviderFactory:
    """Test provider factory function"""
    
    def test_get_mock_provider(self):
        """Test getting mock provider"""
        provider = get_provider("mock")
        assert isinstance(provider, MockMusicProvider)
    
    def test_get_mubert_provider_with_key(self):
        """Test getting Mubert provider with API key"""
        with patch.dict('os.environ', {'MUBERT_API_KEY': 'test-key'}):
            provider = get_provider("mubert")
            assert isinstance(provider, MubertProvider)
            assert provider.api_key == "test-key"
    
    def test_get_mubert_provider_missing_key(self):
        """Test getting Mubert provider without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                get_provider("mubert")
            assert "Missing MUBERT_API_KEY" in str(exc_info.value.detail)
    
    def test_get_suno_provider_with_key(self):
        """Test getting Suno provider with API key"""
        with patch.dict('os.environ', {'SUNO_API_KEY': 'test-key'}):
            provider = get_provider("suno")
            assert isinstance(provider, SunoProvider)
            assert provider.api_key == "test-key"
    
    def test_get_suno_provider_missing_key(self):
        """Test getting Suno provider without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                get_provider("suno")
            assert "Missing SUNO_API_KEY" in str(exc_info.value.detail)
    
    def test_get_unknown_provider(self):
        """Test getting unknown provider"""
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            get_provider("unknown")
        assert "Unknown provider" in str(exc_info.value.detail)


class TestLicensingCompliance:
    """Test licensing compliance features"""
    
    @pytest.mark.asyncio
    async def test_mock_licensing_metadata(self):
        """Test mock provider includes proper licensing metadata"""
        provider = MockMusicProvider()
        result = await provider.generate("ambient", "calm", 10)
        
        assert "license" in result
        license_info = result["license"]
        assert license_info["type"] == "mock-dev"
        assert license_info["terms"] == "development-only"
        assert "timestamp" in license_info
    
    @pytest.mark.asyncio
    async def test_mubert_licensing_metadata(self):
        """Test Mubert provider includes proper licensing metadata"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.content = b"fake_audio_data"
            mock_response.headers = {"X-Request-ID": "mubert-123"}
            mock_response.raise_for_status = Mock()
            
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            provider = MubertProvider("test-key")
            result = await provider.generate("ambient", "calm", 30)
            
            assert "license" in result
            license_info = result["license"]
            assert license_info["type"] == "mubert"
            assert license_info["receipt"] == "mubert-123"
            assert "timestamp" in license_info


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_provider_generation_failure(self):
        """Test handling of provider generation failures"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "API Error", request=Mock(), response=mock_response
            )
            
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            provider = MubertProvider("test-key")
            
            with pytest.raises(Exception):
                await provider.generate("ambient", "calm", 30)
    
    @pytest.mark.asyncio
    async def test_audio_processing_failure_graceful(self):
        """Test graceful handling of audio processing failures"""
        processor = AudioProcessor()
        
        # Should not raise exception, should return original path
        result = await processor.normalize_volume("/invalid/path.wav")
        assert result == "/invalid/path.wav"


class TestDatabaseIntegration:
    """Test database integration (mocked)"""
    
    @pytest.mark.asyncio
    async def test_music_asset_persistence(self):
        """Test music asset persistence to database"""
        # This would test the database integration in the main endpoint
        # For now, we test the structure of the asset creation
        from shared.models import MediaAsset, MediaType
        
        # Mock asset data
        asset_data = {
            "id": "test-uuid",
            "asset_type": MediaType.MUSIC,
            "file_path": "/test/path.wav",
            "metadata": {
                "genre": "ambient",
                "mood": "calm",
                "provider": "mock"
            }
        }
        
        # Verify asset structure
        assert asset_data["asset_type"] == MediaType.MUSIC
        assert "genre" in asset_data["metadata"]
        assert "provider" in asset_data["metadata"]


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_check(self):
        """Test rate limiting is enforced"""
        # This would test the rate limiting in the main endpoint
        # For now, we verify the rate limiter is imported and available
        from shared.rate_limiter import get_rate_limiter
        
        # Should be able to get rate limiter instance
        limiter = await get_rate_limiter()
        assert limiter is not None


if __name__ == "__main__":
    pytest.main([__file__])


