"""
Base provider class for image generation services
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseImageProvider(ABC):
    """Base class for all image generation providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a single image from a text prompt
        
        Args:
            prompt: Text description of the image to generate
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary containing generation result with keys:
            - success: bool indicating if generation was successful
            - image_url or image_data: Generated image location or data
            - provider: Name of the provider used
            - metadata: Additional generation metadata
        """
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate multiple images from text prompts
        
        Args:
            prompts: List of text descriptions
            **kwargs: Provider-specific parameters
            
        Returns:
            List of generation results, one per prompt
        """
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about provider capabilities
        
        Returns:
            Dictionary with provider information
        """
        return {
            "name": self.name,
            "features": [],
            "supported_formats": ["png", "jpg"],
            "max_batch_size": 10
        }
    
    async def validate_prompt(self, prompt: str) -> bool:
        """
        Validate if a prompt is acceptable for this provider
        
        Args:
            prompt: Text prompt to validate
            
        Returns:
            True if prompt is valid, False otherwise
        """
        if not prompt or not prompt.strip():
            return False
        
        # Basic length check
        if len(prompt) > 4000:
            return False
            
        return True
    
    def optimize_prompt(self, prompt: str, **kwargs) -> str:
        """
        Optimize prompt for better generation results
        
        Args:
            prompt: Original prompt
            **kwargs: Optimization parameters
            
        Returns:
            Optimized prompt
        """
        # Default implementation just returns the original prompt
        return prompt.strip()