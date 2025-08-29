import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock
from services.tts_service.main import app
from services.tts_service.providers.openai_provider import OpenAITTSProvider
from services.tts_service.providers.elevenlabs_provider import ElevenLabsProvider
from services.tts_service.providers.azure_provider import AzureTTSProvider
from services.tts_service.audio_processor import AudioProcessor

@pytest.fixture
def client():
    """Test client for TTS service"""
    return httpx.AsyncClient(app=app, base_url="http://test")

@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check endpoint"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "tts-service"

@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "TTS Service API" in data["message"]
    assert data["version"] == "1.0.0"

@pytest.mark.asyncio
async def test_providers_endpoint(client):
    """Test providers endpoint"""
    response = await client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert len(data["providers"]) == 3
    
    provider_names = [p["name"] for p in data["providers"]]
    assert "elevenlabs" in provider_names
    assert "openai" in provider_names
    assert "azure" in provider_names

@pytest.mark.asyncio
async def test_voices_endpoint(client):
    """Test voices endpoint"""
    response = await client.get("/voices/available")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # Should at least have OpenAI voices
    openai_voices = [v for v in data if v["provider"] == "openai"]
    assert len(openai_voices) > 0
    
    # Check voice structure
    if openai_voices:
        voice = openai_voices[0]
        assert "id" in voice
        assert "name" in voice
        assert "provider" in voice
        assert "language" in voice

@pytest.mark.asyncio
async def test_synthesize_speech_validation(client):
    """Test TTS synthesis input validation"""
    # Test missing text
    response = await client.post("/synthesize/speech", json={})
    assert response.status_code == 422
    
    # Test invalid speed
    response = await client.post("/synthesize/speech", json={
        "text": "Hello world",
        "speed": 3.0  # Too high
    })
    assert response.status_code == 422
    
    # Test invalid pitch
    response = await client.post("/synthesize/speech", json={
        "text": "Hello world",
        "pitch": 0.1  # Too low
    })
    assert response.status_code == 422
    
    # Test text too long
    response = await client.post("/synthesize/speech", json={
        "text": "x" * 6000  # Too long
    })
    assert response.status_code == 422

class TestOpenAIProvider:
    """Test OpenAI TTS provider"""
    
    def test_init_without_api_key(self):
        """Test provider initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):
            provider = OpenAITTSProvider()
            assert provider.client is None
    
    def test_init_with_api_key(self):
        """Test provider initialization with API key"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAITTSProvider()
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_get_voices(self):
        """Test getting available voices"""
        provider = OpenAITTSProvider()
        voices = await provider.get_voices()
        
        assert len(voices) == 6  # OpenAI has 6 predefined voices
        voice_ids = [v.id for v in voices]
        assert "alloy" in voice_ids
        assert "echo" in voice_ids
        assert "fable" in voice_ids
        assert "onyx" in voice_ids
        assert "nova" in voice_ids
        assert "shimmer" in voice_ids
    
    def test_get_capabilities(self):
        """Test provider capabilities"""
        provider = OpenAITTSProvider()
        capabilities = provider.get_capabilities()
        
        assert "formats" in capabilities
        assert "mp3" in capabilities["formats"]
        assert capabilities["speed_control"] is True
        assert capabilities["pitch_control"] is False
        assert capabilities["max_characters"] == 4096

class TestElevenLabsProvider:
    """Test ElevenLabs TTS provider"""
    
    def test_init_without_api_key(self):
        """Test provider initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):
            provider = ElevenLabsProvider()
            assert provider.api_key is None
    
    def test_init_with_api_key(self):
        """Test provider initialization with API key"""
        with patch.dict('os.environ', {'ELEVENLABS_API_KEY': 'test-key'}):
            provider = ElevenLabsProvider()
            assert provider.api_key == 'test-key'
    
    def test_get_capabilities(self):
        """Test provider capabilities"""
        provider = ElevenLabsProvider()
        capabilities = provider.get_capabilities()
        
        assert "formats" in capabilities
        assert "mp3" in capabilities["formats"]
        assert capabilities["voice_cloning"] is True
        assert capabilities["emotion_control"] is True
        assert capabilities["max_characters"] == 5000

class TestAzureProvider:
    """Test Azure TTS provider"""
    
    def test_init_without_api_key(self):
        """Test provider initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):
            provider = AzureTTSProvider()
            assert provider.api_key is None
    
    def test_init_with_api_key(self):
        """Test provider initialization with API key"""
        with patch.dict('os.environ', {'AZURE_SPEECH_KEY': 'test-key'}):
            provider = AzureTTSProvider()
            assert provider.api_key == 'test-key'
    
    def test_get_capabilities(self):
        """Test provider capabilities"""
        provider = AzureTTSProvider()
        capabilities = provider.get_capabilities()
        
        assert "formats" in capabilities
        assert "mp3" in capabilities["formats"]
        assert "wav" in capabilities["formats"]
        assert capabilities["speed_control"] is True
        assert capabilities["pitch_control"] is True
        assert capabilities["neural_voices"] is True
        assert capabilities["ssml_support"] is True

class TestAudioProcessor:
    """Test audio processing utilities"""
    
    def test_init(self):
        """Test audio processor initialization"""
        processor = AudioProcessor()
        assert processor is not None
    
    @pytest.mark.asyncio
    async def test_get_duration_invalid_file(self):
        """Test getting duration of non-existent file"""
        processor = AudioProcessor()
        duration = await processor.get_duration("/nonexistent/file.mp3")
        assert duration == 0.0
    
    @pytest.mark.asyncio
    async def test_adjust_volume_invalid_data(self):
        """Test volume adjustment with invalid data"""
        processor = AudioProcessor()
        invalid_data = b"invalid audio data"
        result = await processor.adjust_volume(invalid_data, 1.5)
        # Should return original data on error
        assert result == invalid_data

@pytest.mark.asyncio
async def test_tts_synthesis_no_providers(client):
    """Test TTS synthesis when all providers fail"""
    with patch('services.tts_service.main.PROVIDER_FALLBACK_ORDER', []):
        response = await client.post("/synthesize/speech", json={
            "text": "Hello world"
        })
        assert response.status_code == 503

@pytest.mark.asyncio 
async def test_audio_file_not_found(client):
    """Test getting non-existent audio file"""
    response = await client.get("/audio/nonexistent-id")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_delete_audio_file_not_found(client):
    """Test deleting non-existent audio file"""
    response = await client.delete("/audio/nonexistent-id")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_process_audio_file_not_found(client):
    """Test processing non-existent audio file"""
    response = await client.post("/process/audio", json={
        "audio_id": "nonexistent-id",
        "operations": [{"type": "normalize"}]
    })
    assert response.status_code == 404