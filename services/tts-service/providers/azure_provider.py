import os
import httpx
from typing import List, Dict, Any, Optional
from shared.logging import get_logger
from .base_provider import TTSProvider, Voice

logger = get_logger(__name__)

class AzureTTSProvider(TTSProvider):
    """Azure Speech Services TTS provider implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.base_url = f"https://{self.region}.tts.speech.microsoft.com"
        self.client = httpx.AsyncClient(timeout=60.0)
        
        if not self.api_key:
            logger.warning("Azure Speech API key not found in environment")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        format: str = "mp3",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Synthesize speech using Azure Speech Services"""
        
        if not self.api_key:
            raise Exception("Azure Speech API key not configured")
        
        # Default voice if none specified
        if not voice_id:
            voice_id = "en-US-AriaNeural"
        
        # Map format to Azure format
        audio_format = "audio-16khz-128kbitrate-mono-mp3"
        if format == "wav":
            audio_format = "riff-16khz-16bit-mono-pcm"
        elif quality == "premium":
            audio_format = "audio-24khz-160kbitrate-mono-mp3"
        
        # Build SSML with speed and pitch adjustments
        speed_percent = f"{int((speed - 1) * 100):+d}%"
        pitch_percent = f"{int((pitch - 1) * 50):+d}%"
        
        ssml = f"""
        <speak version='1.0' xml:lang='en-US'>
            <voice name='{voice_id}'>
                <prosody rate='{speed_percent}' pitch='{pitch_percent}'>
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        try:
            # Get access token first
            token = await self._get_access_token()
            
            url = f"{self.base_url}/cognitiveservices/v1"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": audio_format,
                "User-Agent": "TTS-Service"
            }
            
            response = await self.client.post(url, headers=headers, content=ssml)
            response.raise_for_status()
            
            audio_data = response.content
            
            return {
                "audio_data": audio_data,
                "voice_used": voice_id,
                "format_used": audio_format,
                "provider": "azure"
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Azure TTS API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Azure TTS synthesis failed: {e.response.text}")
        except Exception as e:
            logger.error(f"Azure TTS synthesis error: {e}")
            raise
    
    async def get_voices(self) -> List[Voice]:
        """Get available voices from Azure Speech Services"""
        
        if not self.api_key:
            return []
        
        try:
            token = await self._get_access_token()
            
            url = f"{self.base_url}/cognitiveservices/voices/list"
            headers = {"Authorization": f"Bearer {token}"}
            
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            voices_data = response.json()
            voices = []
            
            # Filter to neural voices only for better quality
            for voice_data in voices_data:
                if "Neural" in voice_data.get("DisplayName", ""):
                    voice = Voice(
                        id=voice_data["Name"],
                        name=voice_data["DisplayName"],
                        provider="azure",
                        language=voice_data["Locale"],
                        gender=voice_data.get("Gender"),
                        description=f"{voice_data['LocaleName']} - {voice_data.get('StyleList', ['Standard'])[0]}"
                    )
                    voices.append(voice)
            
            return voices[:50]  # Limit to first 50 voices
            
        except Exception as e:
            logger.error(f"Failed to get Azure voices: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Azure Speech Services health"""
        
        if not self.api_key:
            return False
        
        try:
            token = await self._get_access_token()
            return token is not None
            
        except Exception as e:
            logger.error(f"Azure health check failed: {e}")
            return False
    
    async def _get_access_token(self) -> str:
        """Get Azure access token"""
        
        url = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = await self.client.post(url, headers=headers)
        response.raise_for_status()
        
        return response.text
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Azure TTS capabilities"""
        return {
            "formats": ["mp3", "wav"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi", "ko", "sv", "da", "no"],
            "voice_cloning": False,
            "emotion_control": True,
            "speed_control": True,
            "pitch_control": True,
            "max_characters": 10000,
            "quality_levels": ["standard", "premium"],
            "neural_voices": True,
            "ssml_support": True
        }