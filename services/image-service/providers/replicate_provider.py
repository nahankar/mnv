from .base_provider import BaseImageProvider
import replicate
from typing import Dict, Any, List, Optional
import asyncio
import logging
import time
from ..shared.retry import retry_with_backoff
from ..shared.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class ReplicateProvider(BaseImageProvider):
    def __init__(self, api_token: str):
        super().__init__("replicate")
        self.client = replicate.Client(api_token=api_token)
        self.rate_limiter = RateLimiter(max_requests=100, time_window=60)  # 100 requests per minute
        
        # Available models on Replicate
        self.models = {
            "sdxl": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "sd-1.5": "runwayml/stable-diffusion-v1-5:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            "kandinsky": "ai-forever/kandinsky-2.2:ad9d7879fbffa2874e1d909d1d37d9bc682889cc65b31f7bb00d2362619f194a",
            "playground": "playgroundai/playground-v2-1024px-aesthetic:42fe626e41cc811eaf02c94b892774839268ce1994ea778eba97103fe1ef51b8"
        }
        self.default_model = "sdxl"

    def optimize_prompt(self, prompt: str, style: Optional[str] = None, context: Optional[Dict] = None) -> tuple[str, str]:
        """Optimize prompt for Replicate models"""
        positive_prompt = prompt
        negative_prompt = "blurry, low quality, distorted, deformed"
        
        # Add style-specific enhancements
        if style:
            style_mappings = {
                "photorealistic": {
                    "positive": "photorealistic, 8k uhd, professional photography, sharp focus",
                    "negative": "cartoon, anime, painting, sketch, artificial"
                },
                "artistic": {
                    "positive": "artistic masterpiece, detailed illustration, fine art",
                    "negative": "photograph, realistic, plain"
                },
                "fantasy": {
                    "positive": "fantasy art, magical, mystical, ethereal, detailed",
                    "negative": "modern, realistic, plain, boring"
                },
                "anime": {
                    "positive": "anime style, manga, detailed, colorful",
                    "negative": "realistic, photograph, western art"
                }
            }
            if style in style_mappings:
                positive_prompt += f", {style_mappings[style]['positive']}"
                negative_prompt += f", {style_mappings[style]['negative']}"
        
        return positive_prompt, negative_prompt

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a single image with Replicate"""
        await self.rate_limiter.acquire()
        
        try:
            # Get model
            model_name = kwargs.get("model", self.default_model)
            model_id = self.models.get(model_name, self.models[self.default_model])
            
            # Optimize prompts
            positive_prompt, negative_prompt = self.optimize_prompt(
                prompt,
                kwargs.get("image_style"),
                kwargs.get("context")
            )
            
            # Add custom negative prompt if provided
            if kwargs.get("negative_prompt"):
                negative_prompt += f", {kwargs.get('negative_prompt')}"
            
            # Prepare input parameters based on model
            input_params = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "num_inference_steps": kwargs.get("steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "seed": kwargs.get("seed")
            }
            
            # Remove None values
            input_params = {k: v for k, v in input_params.items() if v is not None}
            
            start_time = time.time()
            
            # Run the model
            output = await asyncio.to_thread(
                self.client.run,
                model_id,
                input=input_params
            )
            
            generation_time = time.time() - start_time
            
            if output:
                # Handle different output formats
                if isinstance(output, list):
                    image_url = output[0] if output else None
                    all_urls = output
                else:
                    image_url = output
                    all_urls = [output]
                
                return {
                    "success": True,
                    "image_url": image_url,
                    "image_urls": all_urls,
                    "original_prompt": prompt,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "provider": self.name,
                    "model": model_name,
                    "model_id": model_id,
                    "parameters": input_params,
                    "generation_time": generation_time,
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "error": "No output generated",
                    "error_type": "no_output",
                    "provider": self.name
                }
                
        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Replicate API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "api_error",
                "provider": self.name
            }
        except Exception as e:
            logger.error(f"Replicate generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "unknown",
                "provider": self.name
            }

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple images with proper rate limiting"""
        batch_size = kwargs.get("batch_size", 8)  # Moderate batch size for Replicate
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
                await asyncio.sleep(3)
        
        return results

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider capabilities and limits"""
        return {
            "name": self.name,
            "available_models": list(self.models.keys()),
            "rate_limit": "100 requests/minute",
            "features": [
                "multiple_models",
                "negative_prompts",
                "guidance_scale_control",
                "step_control",
                "seed_control",
                "custom_dimensions"
            ],
            "batch_support": True,
            "style_optimization": True
        }