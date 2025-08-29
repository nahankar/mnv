"""
Mock provider for testing image generation without real API calls
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from .base_provider import BaseImageProvider
import logging

logger = logging.getLogger(__name__)


class MockProvider(BaseImageProvider):
    """Mock image provider for testing and development"""
    
    def __init__(self, fail_rate: float = 0.1):
        super().__init__("mock")
        self.generation_delay = 2.0  # Simulate API call delay
        self.fail_rate = fail_rate  # Configurable failure rate
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a mock image response"""
        # Simulate API call delay
        await asyncio.sleep(self.generation_delay)
        
        # Simulate occasional failures for testing
        import random
        if random.random() < self.fail_rate:
            return {
                "success": False,
                "error": "Mock API error for testing",
                "error_type": "api_error",
                "provider": self.name
            }
        
        # Generate mock response
        image_id = str(uuid.uuid4())
        mock_url = f"https://mock-images.example.com/{image_id}.png"
        
        # Create a simple 1x1 PNG image as base64 data for testing
        import base64
        # This is a 1x1 transparent PNG
        mock_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        
        return {
            "success": True,
            "image_data": mock_image_data,
            "image_id": image_id,
            "original_prompt": prompt,
            "provider": self.name,
            "parameters": kwargs,
            "generation_time": self.generation_delay,
            "timestamp": time.time(),
            "metadata": {
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "format": "png",
                "mock": True
            }
        }
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple mock images"""
        results = []
        for prompt in prompts:
            result = await self.generate_image(prompt, **kwargs)
            results.append(result)
            # Small delay between batch items
            await asyncio.sleep(0.1)
        return results
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get mock provider information"""
        return {
            "name": self.name,
            "type": "mock",
            "features": [
                "batch_support",
                "custom_dimensions",
                "fast_generation"
            ],
            "supported_formats": ["png", "jpg"],
            "max_batch_size": 50,
            "rate_limit": "unlimited",
            "note": "Mock provider for testing and development"
        }