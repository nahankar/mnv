"""
Redis-backed rate limiter for distributed rate limiting across service instances
"""

import asyncio
import time
import logging
from typing import Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Redis-backed rate limiter using sliding window algorithm"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        identifier: Optional[str] = None
    ) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Rate limit key (e.g., "provider:dalle")
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds
            identifier: Optional identifier for per-client limiting
            
        Returns:
            True if request is allowed, False otherwise
        """
        full_key = f"rate_limit:{key}"
        if identifier:
            full_key = f"{full_key}:{identifier}"
        
        now = time.time()
        window_start = now - window_seconds
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove expired entries
        pipe.zremrangebyscore(full_key, 0, window_start)
        
        # Count current requests in window
        pipe.zcard(full_key)
        
        # Add current request
        pipe.zadd(full_key, {str(now): now})
        
        # Set expiration
        pipe.expire(full_key, window_seconds + 1)
        
        try:
            results = await pipe.execute()
            current_count = results[1]  # Count after cleanup
            
            # Check if under limit (subtract 1 because we already added current request)
            return current_count < limit
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fail open - allow request if Redis is down
            return True
    
    async def get_remaining(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        identifier: Optional[str] = None
    ) -> int:
        """Get remaining requests in current window"""
        full_key = f"rate_limit:{key}"
        if identifier:
            full_key = f"{full_key}:{identifier}"
        
        now = time.time()
        window_start = now - window_seconds
        
        try:
            # Clean up expired entries and count
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(full_key, 0, window_start)
            pipe.zcard(full_key)
            results = await pipe.execute()
            
            current_count = results[1]
            return max(0, limit - current_count)
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            return limit  # Assume full capacity if Redis is down
    
    async def reset(self, key: str, identifier: Optional[str] = None):
        """Reset rate limit for a key"""
        full_key = f"rate_limit:{key}"
        if identifier:
            full_key = f"{full_key}:{identifier}"
        
        try:
            await self.redis.delete(full_key)
        except Exception as e:
            logger.error(f"Redis rate limiter reset error: {e}")


class CircuitBreaker:
    """Simple circuit breaker for provider reliability"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.redis = redis_client
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
    
    async def is_available(self, provider: str) -> bool:
        """Check if provider is available (circuit closed)"""
        key = f"circuit_breaker:{provider}"
        
        try:
            data = await self.redis.hgetall(key)
            if not data:
                return True  # No failures recorded
            
            state = data.get("state", "closed")
            if state == "closed":
                return True
            
            if state == "open":
                last_failure = float(data.get("last_failure", 0))
                if time.time() - last_failure > self.recovery_timeout:
                    # Try to transition to half-open
                    await self.redis.hset(key, "state", "half_open")
                    return True
                return False
            
            # Half-open state - allow one request
            return True
            
        except Exception as e:
            logger.error(f"Circuit breaker check error: {e}")
            return True  # Fail open
    
    async def record_success(self, provider: str):
        """Record successful request"""
        key = f"circuit_breaker:{provider}"
        
        try:
            # Reset circuit breaker on success
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Circuit breaker success record error: {e}")
    
    async def record_failure(self, provider: str):
        """Record failed request"""
        key = f"circuit_breaker:{provider}"
        
        try:
            pipe = self.redis.pipeline()
            pipe.hincrby(key, "failures", 1)
            pipe.hset(key, "last_failure", time.time())
            pipe.expire(key, self.recovery_timeout * 2)
            
            results = await pipe.execute()
            failure_count = results[0]
            
            # Open circuit if threshold exceeded
            if failure_count >= self.failure_threshold:
                await self.redis.hset(key, "state", "open")
                logger.warning(f"Circuit breaker opened for provider {provider}")
            
        except Exception as e:
            logger.error(f"Circuit breaker failure record error: {e}")


class ProviderRateLimiter:
    """Combined rate limiter and circuit breaker for providers"""
    
    def __init__(self, redis_client: redis.Redis):
        self.rate_limiter = RedisRateLimiter(redis_client)
        self.circuit_breaker = CircuitBreaker(redis_client)
        
        # Provider-specific limits (requests per minute)
        self.provider_limits = {
            "dalle": 50,
            "stability": 150,
            "replicate": 100,
            "mock": 1000  # High limit for testing
        }
    
    async def check_provider_availability(self, provider: str, user_id: Optional[str] = None) -> bool:
        """Check if provider is available considering both rate limits and circuit breaker"""
        
        # Check circuit breaker first
        if not await self.circuit_breaker.is_available(provider):
            return False
        
        # Check rate limits
        limit = self.provider_limits.get(provider, 100)  # Default limit
        
        # Global provider limit
        global_allowed = await self.rate_limiter.is_allowed(
            f"provider:{provider}",
            limit,
            60  # 1 minute window
        )
        
        if not global_allowed:
            return False
        
        # Per-user limit (if user_id provided)
        if user_id:
            user_limit = min(limit // 10, 10)  # 10% of provider limit or 10, whichever is smaller
            user_allowed = await self.rate_limiter.is_allowed(
                f"user:{provider}",
                user_limit,
                60,
                user_id
            )
            if not user_allowed:
                return False
        
        return True
    
    async def record_provider_success(self, provider: str):
        """Record successful provider call"""
        await self.circuit_breaker.record_success(provider)
    
    async def record_provider_failure(self, provider: str):
        """Record failed provider call"""
        await self.circuit_breaker.record_failure(provider)
    
    async def get_provider_stats(self, provider: str) -> dict:
        """Get provider rate limit and circuit breaker stats"""
        limit = self.provider_limits.get(provider, 100)
        remaining = await self.rate_limiter.get_remaining(f"provider:{provider}", limit, 60)
        available = await self.circuit_breaker.is_available(provider)
        
        return {
            "provider": provider,
            "limit_per_minute": limit,
            "remaining": remaining,
            "available": available
        }