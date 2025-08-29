from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Voice(BaseModel):
    id: str
    name: str
    provider: str
    language: str
    gender: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None

class TTSProvider(ABC):
    """Base class for TTS providers"""
    
    @abstractmethod
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        format: str = "mp3",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Synthesize speech from text"""
        pass
    
    @abstractmethod
    async def get_voices(self) -> List[Voice]:
        """Get available voices"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities"""
        pass