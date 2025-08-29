"""
Simple in-memory rate limiter for image service providers
"""

import asyncio
import time
from typing import Dict, List
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter using sliding window"""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Acquire permission to make a request
        Returns True if allowed, False if rate limited
        """
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            # Rate limited - wait until we can make a request
            if self.requests:
                wait_time = self.requests[0] + self.time_window - now
                if wait_time > 0:
                    logger.info(f"Rate limited, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            return True
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window"""
        now = time.time()
        
        # Count requests in current window
        current_requests = sum(1 for req_time in self.requests 
                             if req_time > now - self.time_window)
        
        return max(0, self.max_requests - current_requests)
    
    def reset(self):
        """Reset the rate limiter"""
        self.requests.clear()


class MultiProviderRateLimiter:
    """Rate limiter that manages multiple providers"""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
    
    def add_provider(self, provider_name: str, max_requests: int, time_window: int):
        """Add a rate limiter for a provider"""
        self.limiters[provider_name] = RateLimiter(max_requests, time_window)
    
    async def acquire(self, provider_name: str) -> bool:
        """Acquire permission for a specific provider"""
        if provider_name not in self.limiters:
            logger.warning(f"No rate limiter configured for provider {provider_name}")
            return True
        
        return await self.limiters[provider_name].acquire()
    
    def get_remaining(self, provider_name: str) -> int:
        """Get remaining requests for a provider"""
        if provider_name not in self.limiters:
            return float('inf')
        
        return self.limiters[provider_name].get_remaining()
    
    def reset(self, provider_name: str):
        """Reset rate limiter for a provider"""
        if provider_name in self.limiters:
            self.limiters[provider_name].reset()