"""FastAPI middleware utilities for request tracing and logging"""
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

from .logging import set_correlation_id, get_logger

logger = get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID to requests for tracing"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Set correlation ID in context
        set_correlation_id(correlation_id)
        
        # Add to request state for access in endpoints
        request.state.correlation_id = correlation_id
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Log request completion
        logger.info(
            f"{request.method} {request.url.path} completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
                "correlation_id": correlation_id
            }
        )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log incoming request
        logger.info(
            f"Incoming {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        response = await call_next(request)
        return response


def add_middleware(app: FastAPI) -> None:
    """Add standard middleware to FastAPI app"""
    
    # Add correlation ID middleware (first, so it runs for all requests)
    app.add_middleware(CorrelationMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Standard middleware added to FastAPI app")