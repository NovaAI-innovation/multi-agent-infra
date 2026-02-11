"""
Rate limiting utilities for LLM providers.

This module provides rate limiting capabilities to prevent hitting API rate limits,
especially for providers with strict quotas like Google Gemini.

Usage:
    from multi_agent_infrastructure.rate_limiter import (
        RateLimiterConfig,
        create_rate_limiter,
        get_gemini_rate_limiter,
    )
    
    # Create a rate limiter with custom settings
    config = RateLimiterConfig(requests_per_second=0.5)
    limiter = create_rate_limiter(config)
    
    # Use with Gemini model
    model = get_model("google:gemini-1.5-flash", rate_limiter=limiter)
    
    # Or use the convenient helper
    model = get_gemini_rate_limiter(requests_per_second=0.5)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.rate_limiters import InMemoryRateLimiter


@dataclass
class RateLimiterConfig:
    """
    Configuration for rate limiting.
    
    Attributes:
        requests_per_second: Maximum number of requests per second.
            Default is 1.0 (1 request per second).
        check_every_n_seconds: How often to check the rate limit.
            Default is 0.1 seconds.
        max_bucket_size: Maximum size of the token bucket.
            Default is 10.
    """
    
    requests_per_second: float = 1.0
    """Maximum number of requests per second."""
    
    check_every_n_seconds: float = 0.1
    """How often to check the rate limit in seconds."""
    
    max_bucket_size: float = 10.0
    """Maximum size of the token bucket."""
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.check_every_n_seconds <= 0:
            raise ValueError("check_every_n_seconds must be positive")
        if self.max_bucket_size <= 0:
            raise ValueError("max_bucket_size must be positive")


def create_rate_limiter(config: Optional[RateLimiterConfig] = None) -> InMemoryRateLimiter:
    """
    Create an InMemoryRateLimiter from configuration.
    
    Args:
        config: RateLimiterConfig instance. If None, uses default values.
        
    Returns:
        Configured InMemoryRateLimiter instance
        
    Example:
        ```python
        # Default rate limiter (1 request per second)
        limiter = create_rate_limiter()
        
        # Custom rate limiter (0.5 requests per second = 1 request every 2 seconds)
        config = RateLimiterConfig(requests_per_second=0.5)
        limiter = create_rate_limiter(config)
        ```
    """
    if config is None:
        config = RateLimiterConfig()
    
    return InMemoryRateLimiter(
        requests_per_second=config.requests_per_second,
        check_every_n_seconds=config.check_every_n_seconds,
        max_bucket_size=config.max_bucket_size,
    )


def create_rate_limiter_from_env() -> Optional[InMemoryRateLimiter]:
    """
    Create a rate limiter from environment variables.
    
    Environment variables:
        - RATE_LIMIT_REQUESTS_PER_SECOND: Requests per second (default: 1.0)
        - RATE_LIMIT_CHECK_EVERY_N_SECONDS: Check interval (default: 0.1)
        - RATE_LIMIT_MAX_BUCKET_SIZE: Max bucket size (default: 10.0)
        - ENABLE_RATE_LIMITING: Set to "true" to enable (default: false for most providers)
    
    Returns:
        InMemoryRateLimiter if rate limiting is enabled, None otherwise
        
    Example:
        ```python
        # Set in .env file:
        # ENABLE_RATE_LIMITING=true
        # RATE_LIMIT_REQUESTS_PER_SECOND=0.5
        
        limiter = create_rate_limiter_from_env()
        if limiter:
            model = get_model("google:gemini-1.5-flash", rate_limiter=limiter)
        ```
    """
    if os.environ.get("ENABLE_RATE_LIMITING", "").lower() != "true":
        return None
    
    config = RateLimiterConfig(
        requests_per_second=float(
            os.environ.get("RATE_LIMIT_REQUESTS_PER_SECOND", "1.0")
        ),
        check_every_n_seconds=float(
            os.environ.get("RATE_LIMIT_CHECK_EVERY_N_SECONDS", "0.1")
        ),
        max_bucket_size=float(
            os.environ.get("RATE_LIMIT_MAX_BUCKET_SIZE", "10.0")
        ),
    )
    
    return create_rate_limiter(config)


# Provider-specific rate limit configurations
# These values are based on typical free tier limits
PROVIDER_RATE_LIMITS = {
    "google": {
        # Google Gemini free tier: 60 requests per minute = 1 per second
        "requests_per_second": 1.0,
        "check_every_n_seconds": 0.1,
        "max_bucket_size": 10.0,
    },
    "openai": {
        # OpenAI tier 1: 3,500 RPM = ~58 per second (very high, rate limiting usually not needed)
        "requests_per_second": 50.0,
        "check_every_n_seconds": 0.01,
        "max_bucket_size": 100.0,
    },
    "anthropic": {
        # Anthropic: Depends on tier, but generally generous
        "requests_per_second": 10.0,
        "check_every_n_seconds": 0.1,
        "max_bucket_size": 20.0,
    },
    "moonshot": {
        # Moonshot AI: Check their current limits
        "requests_per_second": 2.0,
        "check_every_n_seconds": 0.1,
        "max_bucket_size": 10.0,
    },
    "groq": {
        # Groq: Very high rate limits
        "requests_per_second": 30.0,
        "check_every_n_seconds": 0.01,
        "max_bucket_size": 60.0,
    },
    "cohere": {
        # Cohere: Moderate limits
        "requests_per_second": 5.0,
        "check_every_n_seconds": 0.1,
        "max_bucket_size": 15.0,
    },
    "openrouter": {
        # OpenRouter: Varies by model, conservative default
        "requests_per_second": 2.0,
        "check_every_n_seconds": 0.1,
        "max_bucket_size": 10.0,
    },
}


def get_provider_rate_limiter(provider: str) -> Optional[InMemoryRateLimiter]:
    """
    Get a rate limiter configured for a specific provider.
    
    This uses sensible defaults based on typical free tier limits for each provider.
    For Google (Gemini), this is highly recommended to avoid 429 errors.
    
    Args:
        provider: Provider name (e.g., "google", "openai", "anthropic")
        
    Returns:
        InMemoryRateLimiter configured for the provider, or None if provider not found
        
    Example:
        ```python
        # Get rate limiter for Google Gemini
        limiter = get_provider_rate_limiter("google")
        model = get_model("google:gemini-1.5-flash", rate_limiter=limiter)
        ```
    """
    provider = provider.lower()
    
    # Check if rate limiting is explicitly disabled for this provider
    env_var = f"DISABLE_RATE_LIMITING_{provider.upper()}"
    if os.environ.get(env_var, "").lower() == "true":
        return None
    
    # Check for provider-specific rate limit settings
    rps_env = os.environ.get(f"{provider.upper()}_RATE_LIMIT_RPS")
    
    if rps_env:
        # Use environment variable override
        config = RateLimiterConfig(
            requests_per_second=float(rps_env),
            check_every_n_seconds=float(
                os.environ.get(f"{provider.upper()}_RATE_LIMIT_CHECK_INTERVAL", "0.1")
            ),
            max_bucket_size=float(
                os.environ.get(f"{provider.upper()}_RATE_LIMIT_BUCKET_SIZE", "10.0")
            ),
        )
        return create_rate_limiter(config)
    
    # Use default configuration for provider
    if provider in PROVIDER_RATE_LIMITS:
        limits = PROVIDER_RATE_LIMITS[provider]
        config = RateLimiterConfig(
            requests_per_second=limits["requests_per_second"],
            check_every_n_seconds=limits["check_every_n_seconds"],
            max_bucket_size=limits["max_bucket_size"],
        )
        return create_rate_limiter(config)
    
    return None


def get_gemini_rate_limiter(
    requests_per_second: float = 1.0,
) -> InMemoryRateLimiter:
    """
    Get a rate limiter specifically configured for Google Gemini models.
    
    This is a convenience function for the common case of using Gemini models,
    which have strict rate limits on the free tier (60 requests per minute).
    
    Args:
        requests_per_second: Maximum requests per second. 
            Default is 1.0 (60 requests per minute).
            Use 0.5 for 30 requests per minute, etc.
            
    Returns:
        InMemoryRateLimiter configured for Gemini
        
    Example:
        ```python
        # Default: 1 request per second (60/min)
        limiter = get_gemini_rate_limiter()
        model = get_model("google:gemini-1.5-flash", rate_limiter=limiter)
        
        # Slower: 1 request every 2 seconds (30/min)
        limiter = get_gemini_rate_limiter(requests_per_second=0.5)
        model = get_model("google:gemini-1.5-flash", rate_limiter=limiter)
        ```
    """
    return create_rate_limiter(
        RateLimiterConfig(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=10.0,
        )
    )


def apply_rate_limiter_to_config(
    provider: str,
    kwargs: dict[str, Any],
    rate_limiter: Optional[Any] = None,
    auto_apply: bool = True,
) -> dict[str, Any]:
    """
    Apply rate limiting to model configuration kwargs.
    
    This is an internal helper that adds rate limiting to the kwargs dict
    passed to model constructors.
    
    Args:
        provider: The provider name
        kwargs: The kwargs dict to modify
        rate_limiter: Explicit rate limiter to use. If None, may auto-create one.
        auto_apply: Whether to automatically apply provider-specific rate limiting
            when no explicit rate_limiter is provided.
            
    Returns:
        Modified kwargs dict
    """
    # If explicit rate limiter provided, use it
    if rate_limiter is not None:
        kwargs["rate_limiter"] = rate_limiter
        return kwargs
    
    # Check for environment-based rate limiting
    env_limiter = create_rate_limiter_from_env()
    if env_limiter is not None:
        kwargs["rate_limiter"] = env_limiter
        return kwargs
    
    # Auto-apply provider-specific rate limiting if enabled
    if auto_apply:
        # For Google/Gemini, always apply rate limiting by default
        if provider == "google":
            provider_limiter = get_provider_rate_limiter(provider)
            if provider_limiter is not None:
                kwargs["rate_limiter"] = provider_limiter
    
    return kwargs
