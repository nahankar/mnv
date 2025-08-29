"""
Retry utilities with exponential backoff for AI service calls
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple
import random

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base class for errors that should trigger retries"""
    pass


class APIError(RetryableError):
    """API-related errors that should be retried"""
    pass


class NetworkError(RetryableError):
    """Network-related errors that should be retried"""
    pass


class RateLimitError(RetryableError):
    """Rate limit errors that should be retried with longer backoff"""
    pass


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (APIError, NetworkError, RateLimitError)
):
    """
    Decorator for exponential backoff retry logic
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exception types that should trigger retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    # Special handling for rate limit errors
                    if isinstance(e, RateLimitError):
                        delay = max(delay, 30.0)  # Minimum 30s for rate limits
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, fail immediately
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {str(e)}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    # Special handling for rate limit errors
                    if isinstance(e, RateLimitError):
                        delay = max(delay, 30.0)  # Minimum 30s for rate limits
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    import time
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, fail immediately
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {str(e)}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {func.__name__} is now HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {func.__name__} is now HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        import time
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker is now CLOSED")
    
    def _on_failure(self):
        """Handle failed call"""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker is now OPEN after {self.failure_count} failures")


# Convenience decorators with common configurations
def retry_api_call(max_retries: int = 3):
    """Retry decorator specifically for API calls"""
    return exponential_backoff(
        max_retries=max_retries,
        base_delay=2.0,
        max_delay=30.0,
        retryable_exceptions=(APIError, NetworkError, RateLimitError)
    )


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Simple retry decorator with exponential backoff"""
    return exponential_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=(Exception,)
    )


# Helper function to convert HTTP errors to retryable errors
def convert_http_error(response_status: int, error_message: str) -> Exception:
    """Convert HTTP status codes to appropriate exception types"""
    if response_status == 429:
        return RateLimitError(f"Rate limit exceeded: {error_message}")
    elif 500 <= response_status < 600:
        return APIError(f"Server error ({response_status}): {error_message}")
    elif response_status in [408, 502, 503, 504]:
        return NetworkError(f"Network error ({response_status}): {error_message}")
    else:
        return Exception(f"HTTP error ({response_status}): {error_message}")