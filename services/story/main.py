"""
Story Generation Service

FastAPI service for generating story content using multiple LLM providers
with fallback logic, retry mechanisms, and cost tracking.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text

from shared.database import DatabaseManager, get_db_manager
from shared.middleware import CorrelationMiddleware

# Prometheus metrics
REQUEST_COUNT = Counter('story_service_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('story_service_request_duration_seconds', 'Request duration')
PROVIDER_CALLS = Counter('story_service_provider_calls_total', 'Provider API calls', ['provider', 'status'])
GENERATION_DURATION = Histogram('story_service_generation_duration_seconds', 'Story generation duration')
from shared.models import Story, ContentStatus
from shared.schemas import StoryRequest, StoryResponse, MetadataRequest, MetadataResponse
from shared.retry import retry_api_call, APIError, NetworkError, RateLimitError, convert_http_error
from shared.rate_limiter import get_rate_limiter, check_rate_limit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Provider configurations
LLM_PROVIDERS = {
    "gpt-4o": {
        "url": "https://api.openai.com/v1/chat/completions",
        "cost_per_token": 0.00003,  # $0.03 per 1K tokens
        "max_tokens": 4000,
        "priority": 1
    },
    "claude-3.5-sonnet": {
        "url": "https://api.anthropic.com/v1/messages",
        "cost_per_token": 0.000015,  # $0.015 per 1K tokens
        "max_tokens": 4000,
        "priority": 2
    },
    "mistral-large": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "cost_per_token": 0.000008,  # $0.008 per 1K tokens
        "max_tokens": 4000,
        "priority": 3
    }
}

class StoryService:
    """Service for generating stories using multiple LLM providers"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.rate_limits = {}  # Simple in-memory rate limiting
        
    async def generate_story(self, request: StoryRequest) -> StoryResponse:
        """Generate story content with fallback logic"""
        
        # Check rate limits
        await self._check_rate_limit(request.user_id or "anonymous")
        
        # Try providers in priority order
        providers = sorted(LLM_PROVIDERS.items(), key=lambda x: x[1]["priority"])
        
        for provider_name, config in providers:
            try:
                logger.info(f"Attempting story generation with {provider_name}")
                
                story_content = await self._call_llm_provider(
                    provider_name, config, request
                )
                
                # Calculate cost
                estimated_tokens = len(story_content.split()) * 1.3  # Rough estimate
                cost = estimated_tokens * config["cost_per_token"]
                
                # Store story in database
                story = await self._store_story(request, story_content, provider_name, cost)
                
                return StoryResponse(
                    id=story.id,
                    content=story_content,
                    word_count=len(story_content.split()),
                    genre=request.genre,
                    theme=request.theme,
                    provider_used=provider_name,
                    generation_cost=cost,
                    status=ContentStatus.COMPLETED
                )
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {str(e)}")
                continue
        
        # All providers failed
        raise HTTPException(
            status_code=503,
            detail="All LLM providers failed. Please try again later."
        )
    
    async def generate_metadata(self, request: MetadataRequest) -> MetadataResponse:
        """Generate platform-specific metadata for stories"""
        
        platform_prompts = {
            "youtube": {
                "title_prompt": "Create an engaging YouTube title (max 60 chars) for this story:",
                "description_prompt": "Write a YouTube description with SEO keywords:",
                "hashtags_count": 5
            },
            "instagram": {
                "title_prompt": "Create a catchy Instagram caption hook:",
                "description_prompt": "Write an Instagram story description:",
                "hashtags_count": 10
            },
            "tiktok": {
                "title_prompt": "Create a viral TikTok title:",
                "description_prompt": "Write a TikTok description with trending elements:",
                "hashtags_count": 8
            },
            "facebook": {
                "title_prompt": "Create an engaging Facebook post title:",
                "description_prompt": "Write a Facebook description that encourages engagement:",
                "hashtags_count": 3
            }
        }
        
        platform_config = platform_prompts.get(request.platform.lower())
        if not platform_config:
            raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")
        
        # Generate metadata using primary LLM
        try:
            metadata_prompt = f"""
            Story: {request.story_content[:500]}...
            
            Generate:
            1. {platform_config['title_prompt']}
            2. {platform_config['description_prompt']}
            3. {platform_config['hashtags_count']} relevant hashtags
            
            Format as JSON:
            {{
                "title": "...",
                "description": "...",
                "hashtags": ["#tag1", "#tag2", ...]
            }}
            """
            
            provider_name = "gpt-4o"  # Use primary provider for metadata
            config = LLM_PROVIDERS[provider_name]
            
            response = await self._call_llm_provider_raw(
                provider_name, config, metadata_prompt
            )
            
            # Parse JSON response
            import json
            metadata = json.loads(response)
            
            return MetadataResponse(
                title=metadata["title"],
                description=metadata["description"],
                hashtags=metadata["hashtags"],
                platform=request.platform
            )
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate metadata"
            )
    
    async def _call_llm_provider(self, provider_name: str, config: Dict, request: StoryRequest) -> str:
        """Call specific LLM provider with story generation prompt"""
        
        # Build story generation prompt
        prompt = self._build_story_prompt(request)
        
        if provider_name == "gpt-4o":
            return await self._call_openai(config, prompt)
        elif provider_name == "claude-3.5-sonnet":
            return await self._call_anthropic(config, prompt)
        elif provider_name == "mistral-large":
            return await self._call_mistral(config, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    async def _call_llm_provider_raw(self, provider_name: str, config: Dict, prompt: str) -> str:
        """Call LLM provider with raw prompt"""
        
        if provider_name == "gpt-4o":
            return await self._call_openai(config, prompt)
        elif provider_name == "claude-3.5-sonnet":
            return await self._call_anthropic(config, prompt)
        elif provider_name == "mistral-large":
            return await self._call_mistral(config, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def _build_story_prompt(self, request: StoryRequest) -> str:
        """Build story generation prompt based on request parameters"""
        
        prompt = f"""
        Write an engaging story with the following specifications:
        
        Genre: {request.genre or 'Any'}
        Theme: {request.theme or 'Creative and interesting'}
        Target length: {request.target_length or '300-500'} words
        Tone: {request.tone or 'Engaging and accessible'}
        
        Requirements:
        - Create a complete story with beginning, middle, and end
        - Make it suitable for video narration (clear, flowing narrative)
        - Include vivid descriptions that can be visualized
        - Keep language appropriate for general audiences
        - Aim for {request.target_length or '300-500'} words
        
        Story:
        """
        
        return prompt.strip()
    
    @retry_api_call(max_retries=3)
    async def _call_openai(self, config: Dict, prompt: str) -> str:
        """Call OpenAI GPT-4o API with retry logic"""
        
        headers = {
            "Authorization": f"Bearer {self._get_api_key('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a creative storyteller who writes engaging, visual stories perfect for video content."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": config["max_tokens"],
            "temperature": 0.8
        }
        
        try:
            response = await self.http_client.post(
                config["url"], 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except httpx.HTTPStatusError as e:
            raise convert_http_error(e.response.status_code, str(e))
        except httpx.RequestError as e:
            raise NetworkError(f"Network error calling OpenAI: {str(e)}")
    
    @retry_api_call(max_retries=3)
    async def _call_anthropic(self, config: Dict, prompt: str) -> str:
        """Call Anthropic Claude API with retry logic"""
        
        headers = {
            "x-api-key": self._get_api_key('ANTHROPIC_API_KEY'),
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": config["max_tokens"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8
        }
        
        try:
            response = await self.http_client.post(
                config["url"],
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["content"][0]["text"].strip()
            
        except httpx.HTTPStatusError as e:
            raise convert_http_error(e.response.status_code, str(e))
        except httpx.RequestError as e:
            raise NetworkError(f"Network error calling Anthropic: {str(e)}")
    
    @retry_api_call(max_retries=3)
    async def _call_mistral(self, config: Dict, prompt: str) -> str:
        """Call Mistral AI API with retry logic"""
        
        headers = {
            "Authorization": f"Bearer {self._get_api_key('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": "You are a creative storyteller who writes engaging, visual stories perfect for video content."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": config["max_tokens"],
            "temperature": 0.8
        }
        
        try:
            response = await self.http_client.post(
                config["url"],
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except httpx.HTTPStatusError as e:
            raise convert_http_error(e.response.status_code, str(e))
        except httpx.RequestError as e:
            raise NetworkError(f"Network error calling Mistral: {str(e)}")
    
    def _get_api_key(self, key_name: str) -> str:
        """Get API key from environment or secrets manager"""
        import os
        api_key = os.getenv(key_name)
        if not api_key:
            raise ValueError(f"Missing API key: {key_name}")
        return api_key
    
    async def _check_rate_limit(self, user_id: str):
        """Check rate limits for story generation"""
        allowed = await check_rate_limit(user_id, "story_generation")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
    
    async def _store_story(self, request: StoryRequest, content: str, provider: str, cost: float) -> Story:
        """Store generated story in database"""
        
        story_data = {
            "content": content,
            "genre": request.genre,
            "theme": request.theme,
            "word_count": len(content.split()),
            "generation_params": {
                "provider": provider,
                "target_length": request.target_length,
                "tone": request.tone,
                "cost": cost
            },
            "status": ContentStatus.COMPLETED
        }
        
        async with self.db_manager.get_session() as session:
            story = Story(**story_data)
            session.add(story)
            await session.commit()
            await session.refresh(story)
            return story

# Global service instance
story_service: Optional[StoryService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global story_service
    
    # Initialize database and service
    db_manager = DatabaseManager()
    await db_manager.initialize()
    story_service = StoryService(db_manager)
    
    logger.info("Story service initialized")
    yield
    
    # Cleanup
    if story_service:
        await story_service.http_client.aclose()
    await db_manager.close()
    logger.info("Story service shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Story Generation Service",
    description="AI-powered story generation with multiple LLM providers",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CorrelationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with structured logging"""
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="Internal server error")

def get_story_service() -> StoryService:
    """Dependency to get story service instance"""
    if story_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return story_service

@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint with optional deep checks"""
    health_status = {"status": "healthy", "service": "story-generation"}
    
    if deep:
        # Deep health check - test database and Redis connectivity
        try:
            db_manager = get_db_manager()
            async with db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
            health_status["database"] = "connected"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["database"] = f"error: {str(e)}"
        
        try:
            # Test Redis connectivity if available
            import redis.asyncio as redis
            redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            await redis_client.ping()
            health_status["redis"] = "connected"
            await redis_client.close()
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate/story", response_model=StoryResponse)
async def generate_story_endpoint(
    request: StoryRequest,
    background_tasks: BackgroundTasks,
    service: StoryService = Depends(get_story_service)
):
    """Generate a story using AI with fallback providers"""
    try:
        response = await service.generate_story(request)
        
        # Track usage in background
        background_tasks.add_task(
            track_usage, 
            "story_generation", 
            response.provider_used, 
            response.generation_cost,
            request.user_id or "anonymous"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Story generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/generate/metadata", response_model=MetadataResponse)
async def generate_metadata_endpoint(
    request: MetadataRequest,
    service: StoryService = Depends(get_story_service)
):
    """Generate platform-specific metadata for stories"""
    try:
        return await service.generate_metadata(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/providers")
async def get_available_providers():
    """Get list of available LLM providers"""
    return {
        "providers": [
            {
                "name": name,
                "priority": config["priority"],
                "cost_per_token": config["cost_per_token"],
                "max_tokens": config["max_tokens"]
            }
            for name, config in LLM_PROVIDERS.items()
        ]
    }

async def track_usage(operation: str, provider: str, cost: float, user_id: str = "system"):
    """Track API usage and costs"""
    try:
        limiter = await get_rate_limiter()
        await limiter.track_cost(user_id, operation, cost)
        logger.info(f"Usage tracked: {operation} via {provider}, cost: ${cost:.4f}, user: {user_id}")
    except Exception as e:
        logger.error(f"Failed to track usage: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)