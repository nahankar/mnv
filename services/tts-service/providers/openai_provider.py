import os
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from shared.logging import get_logger
from .base_provider import TTSProvider, Voice

logger = get_logger(__name__)

class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        
        if not self.api_key:
            logger.warning("OpenAI API key not found in environment")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        format: str = "mp3",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Synthesize speech using OpenAI TTS API"""
        
        if not self.client:
            raise Exception("OpenAI API key not configured")
        
        # Default voice if none specified
        if not voice_id:
            voice_id = "alloy"
        
        # Map quality to model
        model = "tts-1"
        if quality == "premium":
            model = "tts-1-hd"
        
        # OpenAI supports limited formats
        response_format = "mp3"
        if format in ["opus", "aac", "flac"]:
            response_format = format
        
        try:
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice_id,
                input=text,
                response_format=response_format,
                speed=max(0.25, min(4.0, speed))  # OpenAI speed range
            )
            
            audio_data = response.content
            
            return {
                "audio_data": audio_data,
                "voice_used": voice_id,
                "model_used": model,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}")
            raise Exception(f"OpenAI TTS synthesis failed: {str(e)}")
    
    async def get_voices(self) -> List[Voice]:
        """Get available voices from OpenAI"""
        
        # OpenAI has predefined voices
        voices = [
            Voice(
                id="alloy",
                name="Alloy",
                provider="openai",
                language="en",
                description="Neutral, balanced voice"
            ),
            Voice(
                id="echo",
                name="Echo",
                provider="openai",
                language="en",
                description="Clear, articulate voice"
            ),
            Voice(
                id="fable",
                name="Fable",
                provider="openai",
                language="en",
                description="Warm, storytelling voice"
            ),
            Voice(
                id="onyx",
                name="Onyx",
                provider="openai",
                language="en",
                description="Deep, authoritative voice"
            ),
            Voice(
                id="nova",
                name="Nova",
                provider="openai",
                language="en",
                description="Bright, energetic voice"
            ),
            Voice(
                id="shimmer",
                name="Shimmer",
                provider="openai",
                language="en",
                description="Gentle, soothing voice"
            )
        ]
        
        return voices
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        
        if not self.client:
            return False
        
        try:
            # Test with minimal request
            response = await self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="test",
                response_format="mp3"
            )
            return len(response.content) > 0
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get OpenAI TTS capabilities"""
        return {
            "formats": ["mp3", "opus", "aac", "flac"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi", "ko"],
            "voice_cloning": False,
            "emotion_control": False,
            "speed_control": True,
            "pitch_control": False,
            "max_characters": 4096,
            "quality_levels": ["standard", "premium"]
        }