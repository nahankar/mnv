from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy import text
import logging
import os

from shared.database import DatabaseManager, get_db_manager
from shared.logging import get_logger
from shared.config import get_config
from shared.middleware import add_middleware
from shared.schemas import StoryRequest, StoryResponse, MetadataRequest, MetadataResponse
from shared.models import Story, ContentStatus
from shared.retry import retry_api_call, convert_http_error, NetworkError
from shared.rate_limiter import get_rate_limiter, check_rate_limit

import httpx

app = FastAPI(title="Story Service", version="1.0.0")
logger = get_logger(__name__)
config = get_config()

# Add standard middleware for correlation IDs and request logging
add_middleware(app)


# LLM Providers configuration
LLM_PROVIDERS = {
    "gpt-4o": {
        "url": "https://api.openai.com/v1/chat/completions",
        "cost_per_token": 0.00003,
        "max_tokens": 4000,
        "priority": 1,
    },
    "claude-3.5-sonnet": {
        "url": "https://api.anthropic.com/v1/messages",
        "cost_per_token": 0.000015,
        "max_tokens": 4000,
        "priority": 2,
    },
    "mistral-large": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "cost_per_token": 0.000008,
        "max_tokens": 4000,
        "priority": 3,
    },
}


class StoryService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.http_client = httpx.AsyncClient(timeout=60.0)

    async def generate_story(self, request: StoryRequest) -> StoryResponse:
        user_id = request.user_id or "anonymous"
        allowed = await check_rate_limit(user_id, "story_generation")
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

        providers = sorted(LLM_PROVIDERS.items(), key=lambda x: x[1]["priority"]) 
        last_error = None
        for provider_name, cfg in providers:
            try:
                content = await self._call_provider(provider_name, cfg, request)
                estimated_tokens = len(content.split()) * 1.3
                cost = estimated_tokens * cfg["cost_per_token"]
                story = await self._store_story(request, content, provider_name, cost)
                return StoryResponse(
                    id=story.id,
                    content=content,
                    word_count=len(content.split()),
                    genre=request.genre,
                    theme=request.theme,
                    provider_used=provider_name,
                    generation_cost=cost,
                    status=ContentStatus.COMPLETED,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        raise HTTPException(status_code=503, detail="All LLM providers failed. Please try again later.")

    async def generate_metadata(self, request: MetadataRequest) -> MetadataResponse:
        platforms = {
            "youtube": {"title_prompt": "Create an engaging YouTube title (max 60 chars) for this story:", "description_prompt": "Write a YouTube description with SEO keywords:", "hashtags_count": 5},
            "instagram": {"title_prompt": "Create a catchy Instagram caption hook:", "description_prompt": "Write an Instagram story description:", "hashtags_count": 10},
            "tiktok": {"title_prompt": "Create a viral TikTok title:", "description_prompt": "Write a TikTok description with trending elements:", "hashtags_count": 8},
            "facebook": {"title_prompt": "Create an engaging Facebook post title:", "description_prompt": "Write a Facebook description that encourages engagement:", "hashtags_count": 3},
        }
        cfg = platforms.get(request.platform.lower())
        if not cfg:
            raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")

        prompt = (
            f"Story: {request.story_content[:500]}...\n\n"
            f"Generate:\n"
            f"1. {cfg['title_prompt']}\n"
            f"2. {cfg['description_prompt']}\n"
            f"3. {cfg['hashtags_count']} relevant hashtags\n\n"
            "Format as JSON:{\n  \"title\": \"...\",\n  \"description\": \"...\",\n  \"hashtags\": [\"#tag1\", \"#tag2\"]\n}"
        )

        try:
            raw = await self._call_provider_raw("gpt-4o", LLM_PROVIDERS["gpt-4o"], prompt)
            import json
            data = json.loads(raw)
            return MetadataResponse(title=data["title"], description=data["description"], hashtags=data["hashtags"], platform=request.platform)
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate metadata")

    def _build_prompt(self, req: StoryRequest) -> str:
        return (
            f"Write an engaging story with the following specifications:\n\n"
            f"Genre: {req.genre or 'Any'}\n"
            f"Theme: {req.theme or 'Creative and interesting'}\n"
            f"Target length: {req.target_length or '300-500'} words\n"
            f"Tone: {req.tone or 'Engaging and accessible'}\n\n"
            "Requirements:\n"
            "- Create a complete story with beginning, middle, and end\n"
            "- Make it suitable for video narration (clear, flowing narrative)\n"
            "- Include vivid descriptions that can be visualized\n"
            f"- Aim for {req.target_length or '300-500'} words\n\n"
            "Story:\n"
        )

    async def _call_provider(self, name: str, cfg: dict, req: StoryRequest) -> str:
        prompt = self._build_prompt(req)
        if name == "gpt-4o":
            return await self._call_openai(cfg, prompt)
        if name == "claude-3.5-sonnet":
            return await self._call_anthropic(cfg, prompt)
        if name == "mistral-large":
            return await self._call_mistral(cfg, prompt)
        raise ValueError(f"Unknown provider: {name}")

    async def _call_provider_raw(self, name: str, cfg: dict, prompt: str) -> str:
        if name == "gpt-4o":
            return await self._call_openai(cfg, prompt)
        if name == "claude-3.5-sonnet":
            return await self._call_anthropic(cfg, prompt)
        if name == "mistral-large":
            return await self._call_mistral(cfg, prompt)
        raise ValueError(f"Unknown provider: {name}")

    def _get_api_key(self, name: str) -> str:
        key = os.getenv(name)
        if not key:
            raise ValueError(f"Missing API key: {name}")
        return key

    @retry_api_call(max_retries=3)
    async def _call_openai(self, cfg: dict, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self._get_api_key('OPENAI_API_KEY')}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a creative storyteller who writes engaging, visual stories perfect for video content."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": cfg["max_tokens"],
            "temperature": 0.8,
        }
        try:
            resp = await self.http_client.post(cfg["url"], headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            raise convert_http_error(e.response.status_code, str(e))
        except httpx.RequestError as e:
            raise NetworkError(f"Network error calling OpenAI: {str(e)}")

    @retry_api_call(max_retries=3)
    async def _call_anthropic(self, cfg: dict, prompt: str) -> str:
        headers = {"x-api-key": self._get_api_key("ANTHROPIC_API_KEY"), "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": cfg["max_tokens"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
        }
        try:
            resp = await self.http_client.post(cfg["url"], headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()
        except httpx.HTTPStatusError as e:
            raise convert_http_error(e.response.status_code, str(e))
        except httpx.RequestError as e:
            raise NetworkError(f"Network error calling Anthropic: {str(e)}")

    @retry_api_call(max_retries=3)
    async def _call_mistral(self, cfg: dict, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self._get_api_key('MISTRAL_API_KEY')}", "Content-Type": "application/json"}
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": "You are a creative storyteller who writes engaging, visual stories perfect for video content."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": cfg["max_tokens"],
            "temperature": 0.8,
        }
        try:
            resp = await self.http_client.post(cfg["url"], headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            raise convert_http_error(e.response.status_code, str(e))
        except httpx.RequestError as e:
            raise NetworkError(f"Network error calling Mistral: {str(e)}")

    async def _store_story(self, request: StoryRequest, content: str, provider: str, cost: float) -> Story:
        payload = {
            "content": content,
            "genre": request.genre,
            "theme": request.theme,
            "word_count": len(content.split()),
            "generation_params": {"provider": provider, "target_length": request.target_length, "tone": request.tone, "cost": cost},
            "status": ContentStatus.COMPLETED,
        }
        async with self.db_manager.get_session() as session:
            story = Story(**payload)
            session.add(story)
            await session.commit()
            await session.refresh(story)
            return story


service: StoryService | None = None


@app.on_event("startup")
async def on_startup():
    global service
    dbm = get_db_manager()
    await dbm.initialize()
    service = StoryService(dbm)
    logger.info("Story service initialized")


@app.on_event("shutdown")
async def on_shutdown():
    global service
    if service:
        await service.http_client.aclose()
    await get_db_manager().close()
    logger.info("Story service shutdown complete")


def get_service() -> StoryService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service


@app.get("/health")
async def health_check(deep: bool = False):
    status = {"status": "healthy", "service": "story-service"}
    if deep:
        try:
            async with get_db_manager().get_session() as session:
                await session.execute(text("SELECT 1"))
            status["database"] = "connected"
        except Exception as e:
            status["status"] = "unhealthy"
            status["database"] = f"error: {str(e)}"
        try:
            import redis.asyncio as redis
            client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            await client.ping()
            status["redis"] = "connected"
            await client.close()
        except Exception as e:
            status["redis"] = f"error: {str(e)}"
    return status


@app.get("/")
async def root():
    return {"message": "Story Service API", "version": "1.0.0"}


@app.post("/generate/story", response_model=StoryResponse)
async def generate_story_endpoint(request: StoryRequest, background_tasks: BackgroundTasks, svc: StoryService = Depends(get_service)):
    try:
        resp = await svc.generate_story(request)
        background_tasks.add_task(track_usage, "story_generation", resp.provider_used, resp.generation_cost, request.user_id or "anonymous")
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/generate/metadata", response_model=MetadataResponse)
async def generate_metadata_endpoint(request: MetadataRequest, svc: StoryService = Depends(get_service)):
    try:
        return await svc.generate_metadata(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/providers")
async def get_providers():
    return {
        "providers": [
            {"name": name, "priority": cfg["priority"], "cost_per_token": cfg["cost_per_token"], "max_tokens": cfg["max_tokens"]}
            for name, cfg in LLM_PROVIDERS.items()
        ]
    }


async def track_usage(operation: str, provider: str, cost: float, user_id: str = "system"):
    try:
        limiter = await get_rate_limiter()
        await limiter.track_cost(user_id, operation, cost)
        logger.info(f"Usage tracked: {operation} via {provider}, cost: ${cost:.4f}, user: {user_id}")
    except Exception as e:
        logger.error(f"Failed to track usage: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)