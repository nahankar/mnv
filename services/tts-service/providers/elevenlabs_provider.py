import os
import httpx
from typing import List, Dict, Any, Optional
from shared.logging import get_logger
from .base_provider import TTSProvider, Voice

logger = get_logger(__name__)

class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS provider implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        self.client = httpx.AsyncClient(timeout=60.0)
        
        if not self.api_key:
            logger.warning("ElevenLabs API key not found in environment")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        format: str = "mp3",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Synthesize speech using ElevenLabs API"""
        
        if not self.api_key:
            raise Exception("ElevenLabs API key not configured")
        
        # Default voice if none specified
        if not voice_id:
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        # Map quality to model
        model_id = "eleven_monolingual_v1"
        if quality == "premium":
            model_id = "eleven_multilingual_v2"
        elif quality == "turbo":
            model_id = "eleven_turbo_v2"
        
        # Prepare request
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        # Adjust voice settings based on speed and pitch
        if speed != 1.0:
            # ElevenLabs doesn't have direct speed control, adjust stability
            data["voice_settings"]["stability"] = max(0.1, min(1.0, 0.5 / speed))
        
        try:
            response = await self.client.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            audio_data = response.content
            
            return {
                "audio_data": audio_data,
                "voice_used": voice_id,
                "model_used": model_id,
                "provider": "elevenlabs"
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"ElevenLabs API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"ElevenLabs synthesis failed: {e.response.text}")
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            raise
    
    async def get_voices(self) -> List[Voice]:
        """Get available voices from ElevenLabs"""
        
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            voices_data = response.json()
            voices = []
            
            for voice_data in voices_data.get("voices", []):
                voice = Voice(
                    id=voice_data["voice_id"],
                    name=voice_data["name"],
                    provider="elevenlabs",
                    language=voice_data.get("labels", {}).get("language", "en"),
                    gender=voice_data.get("labels", {}).get("gender"),
                    description=voice_data.get("description"),
                    preview_url=voice_data.get("preview_url")
                )
                voices.append(voice)
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get ElevenLabs voices: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check ElevenLabs API health"""
        
        if not self.api_key:
            return False
        
        try:
            url = f"{self.base_url}/user"
            headers = {"xi-api-key": self.api_key}
            
            response = await self.client.get(url, headers=headers)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"ElevenLabs health check failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get ElevenLabs capabilities"""
        return {
            "formats": ["mp3"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "hi", "ar", "zh"],
            "voice_cloning": True,
            "emotion_control": True,
            "speed_control": False,  # Limited
            "pitch_control": False,  # Limited
            "max_characters": 5000,
            "quality_levels": ["standard", "premium", "turbo"]
        }