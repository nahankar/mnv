"""
Redis-based rate limiting for API calls
"""

import asyncio
import logging
from typing import Optional
import redis.asyncio as redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter using sliding window"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int = 3600
    ) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Unique identifier for the rate limit (e.g., user_id, service_name)
            limit: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds (default: 1 hour)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        if not self.redis_client:
            logger.warning("Redis client not initialized, allowing request")
            return True
        
        try:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Use Redis sorted set to track requests in sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start.timestamp())
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now.timestamp()): now.timestamp()})
            
            # Set expiration for cleanup
            pipe.expire(key, window_seconds + 60)
            
            results = await pipe.execute()
            current_count = results[1]  # Count after removing old entries
            
            return current_count < limit
            
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            # Fail open - allow request if rate limiter fails
            return True
    
    async def get_remaining(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int = 3600
    ) -> int:
        """Get remaining requests in current window"""
        if not self.redis_client:
            return limit
        
        try:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Remove old entries and count current
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start.timestamp())
            pipe.zcard(key)
            
            results = await pipe.execute()
            current_count = results[1]
            
            return max(0, limit - current_count)
            
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            return limit
    
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Error resetting rate limit: {str(e)}")


class ServiceRateLimiter:
    """Service-specific rate limiter with predefined limits"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.rate_limiter = RateLimiter(redis_url)
        
        # Default rate limits per service per hour
        self.service_limits = {
            "story_generation": 100,  # 100 stories per hour per user
            "metadata_generation": 500,  # 500 metadata requests per hour per user
            "openai_api": 1000,  # 1000 OpenAI API calls per hour
            "anthropic_api": 500,  # 500 Anthropic API calls per hour
            "mistral_api": 800,  # 800 Mistral API calls per hour
        }
    
    async def initialize(self):
        """Initialize rate limiter"""
        await self.rate_limiter.initialize()
    
    async def close(self):
        """Close rate limiter"""
        await self.rate_limiter.close()
    
    async def check_user_limit(self, user_id: str, service: str) -> bool:
        """Check if user is within rate limits for a service"""
        limit = self.service_limits.get(service, 100)
        key = f"user:{user_id}:{service}"
        return await self.rate_limiter.is_allowed(key, limit)
    
    async def check_global_limit(self, service: str) -> bool:
        """Check global rate limits for a service"""
        # Global limits are 10x user limits
        limit = self.service_limits.get(service, 100) * 10
        key = f"global:{service}"
        return await self.rate_limiter.is_allowed(key, limit)
    
    async def get_user_remaining(self, user_id: str, service: str) -> int:
        """Get remaining requests for user"""
        limit = self.service_limits.get(service, 100)
        key = f"user:{user_id}:{service}"
        return await self.rate_limiter.get_remaining(key, limit)
    
    async def track_cost(self, user_id: str, service: str, cost: float):
        """Track API costs for budgeting"""
        try:
            if self.rate_limiter.redis_client:
                key = f"cost:{user_id}:{service}:daily"
                await self.rate_limiter.redis_client.incrbyfloat(key, cost)
                await self.rate_limiter.redis_client.expire(key, 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Error tracking cost: {str(e)}")
    
    async def get_daily_cost(self, user_id: str, service: str) -> float:
        """Get daily cost for user and service"""
        try:
            if self.rate_limiter.redis_client:
                key = f"cost:{user_id}:{service}:daily"
                cost = await self.rate_limiter.redis_client.get(key)
                return float(cost) if cost else 0.0
        except Exception as e:
            logger.error(f"Error getting daily cost: {str(e)}")
            return 0.0


# Global rate limiter instance
rate_limiter: Optional[ServiceRateLimiter] = None


async def get_rate_limiter() -> ServiceRateLimiter:
    """Get global rate limiter instance"""
    global rate_limiter
    if rate_limiter is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        rate_limiter = ServiceRateLimiter(redis_url)
        await rate_limiter.initialize()
    return rate_limiter


async def check_rate_limit(user_id: str, service: str) -> bool:
    """Convenience function to check rate limits"""
    limiter = await get_rate_limiter()
    
    # Check both user and global limits
    user_allowed = await limiter.check_user_limit(user_id, service)
    global_allowed = await limiter.check_global_limit(service)
    
    return user_allowed and global_allowed