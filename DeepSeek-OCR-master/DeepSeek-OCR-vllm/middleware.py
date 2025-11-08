"""
Optional middleware for production deployment
Rate limiting, authentication, etc.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import time
from typing import Dict, Tuple


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware
    Limits requests per IP address
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old requests (older than 1 minute)
        current_time = time.time()
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute"
                }
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Simple API key authentication middleware
    For production, use proper JWT or OAuth2
    """
    
    def __init__(self, app, api_keys: set = None):
        super().__init__(app)
        self.api_keys = api_keys or set()
        # Public endpoints that don't require auth
        self.public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Check for API key
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        
        if api_key:
            # Remove "Bearer " prefix if present
            if api_key.startswith("Bearer "):
                api_key = api_key[7:]
            
            if api_key in self.api_keys:
                return await call_next(request)
        
        # No valid API key
        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized",
                "message": "Valid API key required"
            }
        )


# To use these middlewares, add to api_server.py:
# from middleware import RateLimitMiddleware, AuthMiddleware
# 
# # Rate limiting (60 requests per minute per IP)
# app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
# 
# # Authentication (optional)
# app.add_middleware(AuthMiddleware, api_keys={"your-api-key-here"})

