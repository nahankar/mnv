from .base_provider import BaseImageProvider
import aiohttp
from typing import Dict, Any, List, Optional
import asyncio
import logging
import base64
import io
import time
from PIL import Image
from ..shared.retry import retry_with_backoff
from ..shared.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class StabilityProvider(BaseImageProvider):
    def __init__(self, api_key: str):
        super().__init__("stability")
        self.api_key = api_key
        self.base_url = "https://api.stability.ai"
        self.model = "stable-diffusion-xl-1024-v1-0"
        self.rate_limiter = RateLimiter(max_requests=150, time_window=60)  # 150 requests per minute
        
        # Stability AI specific configurations
        self.supported_models = [
            "stable-diffusion-xl-1024-v1-0",
            "stable-diffusion-v1-6",
            "stable-diffusion-512-v2-1"
        ]
        self.supported_sizes = {
            "square": (1024, 1024),
            "portrait": (832, 1216),
            "landscape": (1216, 832),
            "wide": (1344, 768),
            "tall": (768, 1344)
        }

    def optimize_prompt(self, prompt: str, style: Optional[str] = None, context: Optional[Dict] = None) -> tuple[str, str]:
        """Optimize prompt for Stability AI with negative prompts"""
        positive_prompt = prompt
        negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
        
        # Add style-specific enhancements
        if style:
            style_mappings = {
                "photorealistic": {
                    "positive": "photorealistic, 8k uhd, high resolution, professional photography",
                    "negative": "cartoon, anime, painting, drawing, sketch"
                },
                "artistic": {
                    "positive": "artistic masterpiece, detailed painting, fine art",
                    "negative": "photograph, realistic"
                },
                "fantasy": {
                    "positive": "fantasy art, magical, ethereal, mystical",
                    "negative": "modern, contemporary, realistic"
                },
                "sci-fi": {
                    "positive": "science fiction, futuristic, cyberpunk, high-tech",
                    "negative": "medieval, ancient, primitive"
                }
            }
            if style in style_mappings:
                positive_prompt += f", {style_mappings[style]['positive']}"
                negative_prompt += f", {style_mappings[style]['negative']}"
        
        # Add context-based enhancements
        if context:
            if context.get("mood") == "dark":
                positive_prompt += ", dark atmosphere, moody lighting"
            elif context.get("mood") == "bright":
                positive_prompt += ", bright, cheerful, vibrant colors"
        
        return positive_prompt, negative_prompt

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a single image with Stability AI"""
        await self.rate_limiter.acquire()
        
        try:
            # Get model and size parameters
            model = kwargs.get("model", self.model)
            if model not in self.supported_models:
                model = self.model
            
            size_key = kwargs.get("size", "square")
            width, height = self.supported_sizes.get(size_key, (1024, 1024))
            
            # Optimize prompts
            positive_prompt, negative_prompt = self.optimize_prompt(
                prompt,
                kwargs.get("image_style"),
                kwargs.get("context")
            )
            
            # Add custom negative prompt if provided
            if kwargs.get("negative_prompt"):
                negative_prompt += f", {kwargs.get('negative_prompt')}"
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                data = {
                    "text_prompts": [
                        {"text": positive_prompt, "weight": 1.0},
                        {"text": negative_prompt, "weight": -1.0}
                    ],
                    "cfg_scale": kwargs.get("cfg_scale", 7),
                    "height": height,
                    "width": width,
                    "samples": 1,
                    "steps": kwargs.get("steps", 30),
                    "seed": kwargs.get("seed", 0),
                    "style_preset": kwargs.get("style_preset", "enhance")
                }
                
                async with session.post(
                    f"{self.base_url}/v1/generation/{model}/text-to-image",
                    headers=headers,
                    json=data
                ) as response:
                    generation_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        image_data = base64.b64decode(result["artifacts"][0]["base64"])
                        
                        # Validate image
                        try:
                            img = Image.open(io.BytesIO(image_data))
                            img.verify()
                        except Exception as e:
                            logger.error(f"Generated image validation failed: {e}")
                            return {
                                "success": False,
                                "error": "Invalid image generated",
                                "error_type": "validation_error",
                                "provider": self.name
                            }
                        
                        return {
                            "success": True,
                            "image_data": image_data,
                            "image_format": "png",
                            "original_prompt": prompt,
                            "positive_prompt": positive_prompt,
                            "negative_prompt": negative_prompt,
                            "provider": self.name,
                            "model": model,
                            "parameters": {
                                "width": width,
                                "height": height,
                                "cfg_scale": data["cfg_scale"],
                                "steps": data["steps"],
                                "seed": result["artifacts"][0].get("seed", 0)
                            },
                            "generation_time": generation_time,
                            "timestamp": time.time()
                        }
                    else:
                        error_data = await response.json() if response.content_type == 'application/json' else {"message": await response.text()}
                        logger.error(f"Stability AI API error {response.status}: {error_data}")
                        return {
                            "success": False,
                            "error": error_data.get("message", "Unknown API error"),
                            "error_type": "api_error",
                            "status_code": response.status,
                            "provider": self.name
                        }
                        
        except aiohttp.ClientError as e:
            logger.error(f"Stability AI network error: {e}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "error_type": "network_error",
                "provider": self.name
            }
        except Exception as e:
            logger.error(f"Stability AI generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "unknown",
                "provider": self.name
            }

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple images with proper rate limiting"""
        batch_size = kwargs.get("batch_size", 10)  # Stability AI can handle larger batches
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_tasks = [self.generate_image(prompt, **kwargs) for prompt in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "success": False,
                        "error": str(result),
                        "error_type": "exception",
                        "provider": self.name,
                        "prompt": batch[j]
                    })
                else:
                    results.append(result)
            
            # Add delay between batches
            if i + batch_size < len(prompts):
                await asyncio.sleep(1)
        
        return results

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider capabilities and limits"""
        return {
            "name": self.name,
            "supported_models": self.supported_models,
            "supported_sizes": list(self.supported_sizes.keys()),
            "rate_limit": "150 requests/minute",
            "features": [
                "negative_prompts",
                "style_presets", 
                "cfg_scale_control",
                "step_control",
                "seed_control"
            ],
            "batch_support": True,
            "style_optimization": True
        }