"""
Rate Limiting Middleware for the Ragamala painting generation API.
Provides comprehensive rate limiting with Redis backend, multiple strategies,
and user-specific limits based on roles and API keys.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union

import aioredis
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(__name__)

class RateLimitStrategy:
    """Rate limiting strategy definitions."""
    
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

class RateLimitScope:
    """Rate limiting scope definitions."""
    
    GLOBAL = "global"
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_API_KEY = "per_api_key"
    PER_ENDPOINT = "per_endpoint"

class RateLimitConfig:
    """Rate limit configuration model."""
    
    def __init__(
        self,
        requests: int,
        window: int,
        strategy: str = RateLimitStrategy.FIXED_WINDOW,
        scope: str = RateLimitScope.PER_IP,
        burst_requests: Optional[int] = None,
        burst_window: Optional[int] = None,
        block_duration: Optional[int] = None
    ):
        self.requests = requests
        self.window = window
        self.strategy = strategy
        self.scope = scope
        self.burst_requests = burst_requests or requests * 2
        self.burst_window = burst_window or window // 4
        self.block_duration = block_duration or window * 2

class RateLimitException(Exception):
    """Custom exception for rate limit exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: int,
        limit: int,
        remaining: int = 0,
        reset_time: Optional[datetime] = None
    ):
        self.message = message
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
        super().__init__(message)

class RateLimitInfo:
    """Rate limit information model."""
    
    def __init__(
        self,
        limit: int,
        remaining: int,
        reset_time: datetime,
        retry_after: Optional[int] = None
    ):
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after

class RedisRateLimiter:
    """Redis-based rate limiter implementation."""
    
    def __init__(self, redis_url: str, key_prefix: str = "rate_limit"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis: Optional[aioredis.Redis] = None
        
        # Lua scripts for atomic operations
        self.fixed_window_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        local current_window = math.floor(current_time / window)
        local window_key = key .. ":" .. current_window
        
        local current_count = redis.call('GET', window_key)
        if current_count == false then
            current_count = 0
        else
            current_count = tonumber(current_count)
        end
        
        if current_count >= limit then
            local ttl = redis.call('TTL', window_key)
            if ttl == -1 then
                ttl = window
            end
            return {0, current_count, ttl}
        end
        
        local new_count = redis.call('INCR', window_key)
        if new_count == 1 then
            redis.call('EXPIRE', window_key, window)
        end
        
        local ttl = redis.call('TTL', window_key)
        return {1, new_count, ttl}
        """
        
        self.sliding_window_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        local cutoff = current_time - window
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
        
        -- Count current entries
        local current_count = redis.call('ZCARD', key)
        
        if current_count >= limit then
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if #oldest > 0 then
                retry_after = math.ceil(oldest[2] + window - current_time)
            end
            return {0, current_count, retry_after}
        end
        
        -- Add current request
        redis.call('ZADD', key, current_time, current_time)
        redis.call('EXPIRE', key, window)
        
        local new_count = redis.call('ZCARD', key)
        local retry_after = 0
        
        return {1, new_count, retry_after}
        """
        
        self.token_bucket_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        local tokens_requested = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- Calculate tokens to add
        local time_passed = current_time - last_refill
        local tokens_to_add = math.floor(time_passed * refill_rate)
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        if tokens < tokens_requested then
            local retry_after = math.ceil((tokens_requested - tokens) / refill_rate)
            return {0, tokens, retry_after}
        end
        
        tokens = tokens - tokens_requested
        
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
        redis.call('EXPIRE', key, 3600)  -- 1 hour expiry
        
        return {1, tokens, 0}
        """
    
    async def connect(self):
        """Connect to Redis."""
        if not self.redis:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
    
    def _get_key(self, identifier: str, scope: str) -> str:
        """Generate Redis key for rate limiting."""
        return f"{self.key_prefix}:{scope}:{identifier}"
    
    async def check_rate_limit(
        self,
        identifier: str,
        config: RateLimitConfig,
        endpoint: Optional[str] = None
    ) -> RateLimitInfo:
        """Check rate limit for given identifier."""
        await self.connect()
        
        current_time = time.time()
        key = self._get_key(identifier, config.scope)
        
        if endpoint:
            key = f"{key}:{endpoint}"
        
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            result = await self._check_fixed_window(key, config, current_time)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = await self._check_sliding_window(key, config, current_time)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            result = await self._check_token_bucket(key, config, current_time)
        else:
            raise ValueError(f"Unsupported rate limit strategy: {config.strategy}")
        
        allowed, count, retry_after = result
        
        if not allowed:
            remaining = 0
            reset_time = datetime.fromtimestamp(current_time + retry_after)
            raise RateLimitException(
                message="Rate limit exceeded",
                retry_after=retry_after,
                limit=config.requests,
                remaining=remaining,
                reset_time=reset_time
            )
        
        remaining = max(0, config.requests - count)
        reset_time = datetime.fromtimestamp(current_time + config.window)
        
        return RateLimitInfo(
            limit=config.requests,
            remaining=remaining,
            reset_time=reset_time
        )
    
    async def _check_fixed_window(self, key: str, config: RateLimitConfig, current_time: float):
        """Check fixed window rate limit."""
        result = await self.redis.eval(
            self.fixed_window_script,
            1,
            key,
            config.window,
            config.requests,
            current_time
        )
        return result
    
    async def _check_sliding_window(self, key: str, config: RateLimitConfig, current_time: float):
        """Check sliding window rate limit."""
        result = await self.redis.eval(
            self.sliding_window_script,
            1,
            key,
            config.window,
            config.requests,
            current_time
        )
        return result
    
    async def _check_token_bucket(self, key: str, config: RateLimitConfig, current_time: float):
        """Check token bucket rate limit."""
        refill_rate = config.requests / config.window
        result = await self.redis.eval(
            self.token_bucket_script,
            1,
            key,
            config.requests,  # capacity
            refill_rate,
            current_time,
            1  # tokens requested
        )
        return result
    
    async def reset_rate_limit(self, identifier: str, scope: str, endpoint: Optional[str] = None):
        """Reset rate limit for identifier."""
        await self.connect()
        
        key = self._get_key(identifier, scope)
        if endpoint:
            key = f"{key}:{endpoint}"
        
        await self.redis.delete(key)
    
    async def get_rate_limit_info(
        self,
        identifier: str,
        config: RateLimitConfig,
        endpoint: Optional[str] = None
    ) -> Dict:
        """Get current rate limit information without incrementing."""
        await self.connect()
        
        key = self._get_key(identifier, config.scope)
        if endpoint:
            key = f"{key}:{endpoint}"
        
        current_time = time.time()
        
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            current_window = int(current_time // config.window)
            window_key = f"{key}:{current_window}"
            current_count = await self.redis.get(window_key) or 0
            current_count = int(current_count)
            
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            cutoff = current_time - config.window
            await self.redis.zremrangebyscore(key, '-inf', cutoff)
            current_count = await self.redis.zcard(key)
            
        else:  # Token bucket
            bucket = await self.redis.hmget(key, 'tokens', 'last_refill')
            tokens = int(bucket[0]) if bucket[0] else config.requests
            current_count = config.requests - tokens
        
        remaining = max(0, config.requests - current_count)
        reset_time = datetime.fromtimestamp(current_time + config.window)
        
        return {
            'limit': config.requests,
            'remaining': remaining,
            'reset_time': reset_time.isoformat(),
            'current_count': current_count
        }

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Comprehensive rate limiting middleware."""
    
    def __init__(
        self,
        app,
        redis_url: str,
        default_config: Optional[RateLimitConfig] = None,
        endpoint_configs: Optional[Dict[str, RateLimitConfig]] = None,
        role_configs: Optional[Dict[str, RateLimitConfig]] = None,
        identifier_func: Optional[Callable] = None,
        exempt_paths: Optional[List[str]] = None,
        enable_headers: bool = True
    ):
        super().__init__(app)
        
        self.rate_limiter = RedisRateLimiter(redis_url)
        self.default_config = default_config or RateLimitConfig(
            requests=100,
            window=60,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            scope=RateLimitScope.PER_IP
        )
        self.endpoint_configs = endpoint_configs or {}
        self.role_configs = role_configs or {}
        self.identifier_func = identifier_func or self._default_identifier
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
        self.enable_headers = enable_headers
        
        # In-memory fallback for when Redis is unavailable
        self.memory_store = {}
        self.memory_cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _default_identifier(self, request: Request) -> str:
        """Default identifier function using client IP."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host
    
    def _get_user_role(self, request: Request) -> Optional[str]:
        """Extract user role from request."""
        # Try to get from headers (set by auth middleware)
        user_role = request.headers.get("X-User-Role")
        if user_role:
            return user_role
        
        # Try to get from request state (set by auth dependency)
        if hasattr(request.state, "user") and hasattr(request.state.user, "role"):
            return request.state.user.role
        
        return None
    
    def _get_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None
    
    def _get_rate_limit_config(self, request: Request) -> RateLimitConfig:
        """Get appropriate rate limit configuration for request."""
        path = request.url.path
        method = request.method
        endpoint_key = f"{method}:{path}"
        
        # Check for endpoint-specific config
        if endpoint_key in self.endpoint_configs:
            return self.endpoint_configs[endpoint_key]
        
        # Check for role-specific config
        user_role = self._get_user_role(request)
        if user_role and user_role in self.role_configs:
            return self.role_configs[user_role]
        
        return self.default_config
    
    def _should_exempt(self, request: Request) -> bool:
        """Check if request should be exempt from rate limiting."""
        path = request.url.path
        
        # Check exempt paths
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        
        # Check for admin bypass header
        if request.headers.get("X-Rate-Limit-Bypass") == "admin":
            user_role = self._get_user_role(request)
            if user_role == "admin":
                return True
        
        return False
    
    async def _memory_fallback_check(
        self,
        identifier: str,
        config: RateLimitConfig,
        endpoint: Optional[str] = None
    ) -> RateLimitInfo:
        """Fallback rate limiting using in-memory store."""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.memory_cleanup_interval:
            await self._cleanup_memory_store(current_time)
            self.last_cleanup = current_time
        
        key = f"{identifier}:{config.scope}"
        if endpoint:
            key = f"{key}:{endpoint}"
        
        if key not in self.memory_store:
            self.memory_store[key] = []
        
        # Remove old requests outside the window
        cutoff = current_time - config.window
        self.memory_store[key] = [
            timestamp for timestamp in self.memory_store[key]
            if timestamp > cutoff
        ]
        
        current_count = len(self.memory_store[key])
        
        if current_count >= config.requests:
            oldest_request = min(self.memory_store[key])
            retry_after = int(oldest_request + config.window - current_time)
            
            raise RateLimitException(
                message="Rate limit exceeded (memory fallback)",
                retry_after=max(1, retry_after),
                limit=config.requests,
                remaining=0
            )
        
        # Add current request
        self.memory_store[key].append(current_time)
        
        remaining = config.requests - len(self.memory_store[key])
        reset_time = datetime.fromtimestamp(current_time + config.window)
        
        return RateLimitInfo(
            limit=config.requests,
            remaining=remaining,
            reset_time=reset_time
        )
    
    async def _cleanup_memory_store(self, current_time: float):
        """Clean up expired entries from memory store."""
        keys_to_remove = []
        
        for key, timestamps in self.memory_store.items():
            # Keep only recent timestamps
            recent_timestamps = [
                ts for ts in timestamps
                if current_time - ts < 3600  # Keep last hour
            ]
            
            if recent_timestamps:
                self.memory_store[key] = recent_timestamps
            else:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.memory_store[key]
    
    def _add_rate_limit_headers(self, response: Response, rate_limit_info: RateLimitInfo):
        """Add rate limit headers to response."""
        if not self.enable_headers:
            return
        
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.reset_time.timestamp()))
        
        if rate_limit_info.retry_after:
            response.headers["Retry-After"] = str(rate_limit_info.retry_after)
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method."""
        start_time = time.time()
        
        # Check if request should be exempt
        if self._should_exempt(request):
            response = await call_next(request)
            return response
        
        # Get rate limit configuration
        config = self._get_rate_limit_config(request)
        
        # Get identifier
        identifier = self.identifier_func(request)
        
        # Use API key as identifier if available and scope is per_api_key
        if config.scope == RateLimitScope.PER_API_KEY:
            api_key = self._get_api_key(request)
            if api_key:
                identifier = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Get endpoint for per-endpoint limiting
        endpoint = None
        if config.scope == RateLimitScope.PER_ENDPOINT:
            endpoint = f"{request.method}:{request.url.path}"
        
        try:
            # Check rate limit
            try:
                rate_limit_info = await self.rate_limiter.check_rate_limit(
                    identifier, config, endpoint
                )
            except Exception as e:
                logger.warning(f"Redis rate limiter failed, using memory fallback: {e}")
                rate_limit_info = await self._memory_fallback_check(
                    identifier, config, endpoint
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, rate_limit_info)
            
            # Add processing time header
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except RateLimitException as e:
            logger.warning(
                f"Rate limit exceeded for {identifier}: {e.message}",
                extra={
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "limit": e.limit,
                    "retry_after": e.retry_after
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": e.message,
                    "limit": e.limit,
                    "remaining": e.remaining,
                    "retry_after": e.retry_after,
                    "reset_time": e.reset_time.isoformat() if e.reset_time else None
                },
                headers={
                    "Retry-After": str(e.retry_after),
                    "X-RateLimit-Limit": str(e.limit),
                    "X-RateLimit-Remaining": str(e.remaining),
                    "X-RateLimit-Reset": str(int(e.reset_time.timestamp())) if e.reset_time else "0"
                }
            )
        
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue processing if rate limiting fails
            response = await call_next(request)
            return response

# Utility functions for getting user ID from different sources
def get_user_id_from_token(request: Request) -> Optional[str]:
    """Extract user ID from JWT token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    try:
        # This would typically decode the JWT token
        # For now, return a placeholder
        return request.headers.get("X-User-ID")
    except Exception:
        return None

def get_user_id_from_api_key(request: Request) -> Optional[str]:
    """Extract user ID from API key."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Hash the API key for privacy
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return None

def get_user_id_from_session(request: Request) -> Optional[str]:
    """Extract user ID from session."""
    session_id = request.cookies.get("session_id")
    if session_id:
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return None

# Rate limit configuration presets
class RateLimitPresets:
    """Predefined rate limit configurations."""
    
    # API endpoint limits
    GENERATE_IMAGE = RateLimitConfig(
        requests=10,
        window=60,
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        scope=RateLimitScope.PER_API_KEY
    )
    
    BATCH_GENERATE = RateLimitConfig(
        requests=2,
        window=60,
        strategy=RateLimitStrategy.FIXED_WINDOW,
        scope=RateLimitScope.PER_API_KEY
    )
    
    HEALTH_CHECK = RateLimitConfig(
        requests=100,
        window=60,
        strategy=RateLimitStrategy.FIXED_WINDOW,
        scope=RateLimitScope.PER_IP
    )
    
    # Role-based limits
    ADMIN_LIMITS = RateLimitConfig(
        requests=1000,
        window=60,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        scope=RateLimitScope.PER_USER
    )
    
    USER_LIMITS = RateLimitConfig(
        requests=100,
        window=60,
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        scope=RateLimitScope.PER_USER
    )
    
    ANONYMOUS_LIMITS = RateLimitConfig(
        requests=20,
        window=60,
        strategy=RateLimitStrategy.FIXED_WINDOW,
        scope=RateLimitScope.PER_IP
    )

# Factory function for creating configured middleware
def create_rate_limiting_middleware(
    redis_url: str,
    environment: str = "production"
) -> RateLimitingMiddleware:
    """Factory function to create configured rate limiting middleware."""
    
    # Environment-specific configurations
    if environment == "development":
        default_config = RateLimitConfig(requests=1000, window=60)
        endpoint_configs = {}
        role_configs = {}
    
    elif environment == "staging":
        default_config = RateLimitPresets.USER_LIMITS
        endpoint_configs = {
            "POST:/generate": RateLimitConfig(requests=20, window=60),
            "POST:/batch-generate": RateLimitConfig(requests=5, window=60)
        }
        role_configs = {
            "admin": RateLimitPresets.ADMIN_LIMITS,
            "user": RateLimitPresets.USER_LIMITS
        }
    
    else:  # production
        default_config = RateLimitPresets.ANONYMOUS_LIMITS
        endpoint_configs = {
            "POST:/generate": RateLimitPresets.GENERATE_IMAGE,
            "POST:/batch-generate": RateLimitPresets.BATCH_GENERATE,
            "GET:/health": RateLimitPresets.HEALTH_CHECK
        }
        role_configs = {
            "admin": RateLimitPresets.ADMIN_LIMITS,
            "developer": RateLimitConfig(requests=500, window=60),
            "user": RateLimitPresets.USER_LIMITS,
            "viewer": RateLimitConfig(requests=50, window=60)
        }
    
    return RateLimitingMiddleware(
        app=None,  # Will be set by FastAPI
        redis_url=redis_url,
        default_config=default_config,
        endpoint_configs=endpoint_configs,
        role_configs=role_configs,
        exempt_paths=["/health", "/docs", "/redoc", "/openapi.json", "/metrics"]
    )
