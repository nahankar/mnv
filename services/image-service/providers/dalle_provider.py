from .base_provider import BaseImageProvider
import openai
from typing import Dict, Any, List, Optional
import asyncio
import logging
import time
from ..shared.retry import retry_with_backoff
from ..shared.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class DALLEProvider(BaseImageProvider):
    def __init__(self, api_key: str):
        super().__init__("dalle")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = "dall-e-3"
        self.rate_limiter = RateLimiter(max_requests=50, time_window=60)  # 50 requests per minute
        
        # DALL-E 3 specific configurations
        self.supported_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        self.supported_qualities = ["standard", "hd"]
        self.supported_styles = ["vivid", "natural"]

    def optimize_prompt(self, prompt: str, style: Optional[str] = None, context: Optional[Dict] = None) -> str:
        """Optimize prompt for DALL-E 3 with style and context awareness"""
        optimized = prompt
        
        # Add style guidance
        if style:
            style_mappings = {
                "photorealistic": "photorealistic, highly detailed, professional photography",
                "artistic": "artistic illustration, creative, expressive",
                "cartoon": "cartoon style, animated, colorful",
                "fantasy": "fantasy art, magical, ethereal",
                "sci-fi": "science fiction, futuristic, high-tech"
            }
            if style in style_mappings:
                optimized = f"{optimized}, {style_mappings[style]}"
        
        # Add context from story if available
        if context and context.get("genre"):
            genre = context["genre"].lower()
            if "horror" in genre:
                optimized += ", dark atmosphere, mysterious"
            elif "romance" in genre:
                optimized += ", warm lighting, romantic atmosphere"
            elif "adventure" in genre:
                optimized += ", dynamic, exciting, adventurous"
        
        # Ensure prompt length is appropriate for DALL-E 3
        if len(optimized) > 4000:
            optimized = optimized[:4000]
        
        return optimized

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a single image with DALL-E 3"""
        await self.rate_limiter.acquire()
        
        try:
            # Validate and set parameters
            size = kwargs.get("size", "1024x1024")
            if size not in self.supported_sizes:
                size = "1024x1024"
            
            quality = kwargs.get("quality", "standard")
            if quality not in self.supported_qualities:
                quality = "standard"
            
            style = kwargs.get("style", "vivid")
            if style not in self.supported_styles:
                style = "vivid"
            
            # Optimize prompt
            optimized_prompt = self.optimize_prompt(
                prompt, 
                kwargs.get("image_style"), 
                kwargs.get("context")
            )
            
            start_time = time.time()
            response = await self.client.images.generate(
                model=self.model,
                prompt=optimized_prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            generation_time = time.time() - start_time
            
            result = {
                "success": True,
                "image_url": response.data[0].url,
                "revised_prompt": response.data[0].revised_prompt,
                "original_prompt": prompt,
                "optimized_prompt": optimized_prompt,
                "provider": self.name,
                "model": self.model,
                "parameters": {
                    "size": size,
                    "quality": quality,
                    "style": style
                },
                "generation_time": generation_time,
                "timestamp": time.time()
            }
            
            logger.info(f"DALL-E image generated successfully in {generation_time:.2f}s")
            return result
            
        except openai.RateLimitError as e:
            logger.warning(f"DALL-E rate limit hit: {e}")
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "error_type": "rate_limit",
                "provider": self.name,
                "retry_after": getattr(e, 'retry_after', 60)
            }
        except openai.APIError as e:
            logger.error(f"DALL-E API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "api_error",
                "provider": self.name
            }
        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "unknown",
                "provider": self.name
            }

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple images with proper rate limiting"""
        batch_size = kwargs.get("batch_size", 5)  # Process in smaller batches
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
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(prompts):
                await asyncio.sleep(2)
        
        return results

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider capabilities and limits"""
        return {
            "name": self.name,
            "model": self.model,
            "supported_sizes": self.supported_sizes,
            "supported_qualities": self.supported_qualities,
            "supported_styles": self.supported_styles,
            "rate_limit": "50 requests/minute",
            "max_prompt_length": 4000,
            "batch_support": True,
            "style_optimization": True
        }