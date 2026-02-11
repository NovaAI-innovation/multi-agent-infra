"""
Example: Using Rate Limiting with Google Gemini Models

This example demonstrates how to use rate limiting to avoid hitting API
rate limits when using Google Gemini models (which have a 60 requests/minute
limit on the free tier).

Usage:
    python -m multi_agent_infrastructure.examples.rate_limiter_example
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agent_infrastructure import (
    get_model,
    LLMConfig,
    get_gemini_rate_limiter,
    RateLimiterConfig,
    create_rate_limiter,
)
from langchain_core.messages import HumanMessage


def example_1_basic_gemini_with_rate_limiting():
    """
    Example 1: Basic Gemini model with automatic rate limiting.
    
    When using Google/Gemini models, rate limiting is automatically applied
    by default to prevent hitting the 60 requests/minute free tier limit.
    """
    print("=" * 60)
    print("Example 1: Basic Gemini with Automatic Rate Limiting")
    print("=" * 60)
    
    try:
        # This will automatically apply rate limiting (1 request per second)
        # because Google/Gemini models have strict rate limits
        model = get_model("google:gemini-1.5-flash")
        
        print(f"Model created: {model}")
        print("Rate limiting is automatically applied for Gemini models!")
    except ImportError as e:
        print(f"Note: {e}")
        print("This example requires 'langchain-google-genai' package.")
        print("Install with: pip install langchain-google-genai")
        print()
        print("Code example:")
        print("  model = get_model('google:gemini-1.5-flash')")
        print("  # Rate limiting is automatically applied!")
    print()


def example_2_explicit_rate_limiter():
    """
    Example 2: Create a model with an explicit rate limiter.
    
    You can create a custom rate limiter and pass it to get_model().
    """
    print("=" * 60)
    print("Example 2: Explicit Rate Limiter")
    print("=" * 60)
    
    # Create a rate limiter: 0.5 requests per second = 1 request every 2 seconds
    # This is useful if you want to be extra conservative with API usage
    limiter = get_gemini_rate_limiter(requests_per_second=0.5)
    
    try:
        # Use the rate limiter with the model
        model = get_model(
            "google:gemini-1.5-flash",
            rate_limiter=limiter,
        )
        
        print(f"Model created with custom rate limiter (0.5 RPS)")
        print(f"This allows 1 request every 2 seconds (30 requests/minute)")
    except ImportError as e:
        print(f"Note: {e}")
        print("This example requires 'langchain-google-genai' package.")
        print()
        print("Code example:")
        print("  limiter = get_gemini_rate_limiter(requests_per_second=0.5)")
        print("  model = get_model('google:gemini-1.5-flash', rate_limiter=limiter)")
    print()


def example_3_using_llm_config():
    """
    Example 3: Using LLMConfig for more control.
    
    LLMConfig allows you to configure all aspects of the model including
    rate limiting, temperature, max_tokens, etc.
    """
    print("=" * 60)
    print("Example 3: Using LLMConfig")
    print("=" * 60)
    
    # Create a custom rate limiter
    rate_limiter = create_rate_limiter(
        RateLimiterConfig(
            requests_per_second=1.0,  # 60 requests per minute
            check_every_n_seconds=0.1,
            max_bucket_size=10.0,
        )
    )
    
    # Create configuration
    config = LLMConfig(
        provider="google",
        model="gemini-1.5-flash",
        temperature=0.7,
        max_tokens=1024,
        rate_limiter=rate_limiter,
        auto_rate_limiting=True,  # Enable automatic rate limiting
    )
    
    try:
        # Get the model
        model = config.get_model()
        
        print(f"Model created via LLMConfig")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Rate limiting: Enabled")
    except ImportError as e:
        print(f"Note: {e}")
        print("This example requires 'langchain-google-genai' package.")
        print()
        print("Code example:")
        print("  config = LLMConfig(")
        print("      provider='google',")
        print("      model='gemini-1.5-flash',")
        print("      rate_limiter=rate_limiter,")
        print("  )")
        print("  model = config.get_model()")
    print()


def example_4_disable_auto_rate_limiting():
    """
    Example 4: Disabling automatic rate limiting.
    
    If you have a paid tier with higher limits, you can disable
    automatic rate limiting.
    """
    print("=" * 60)
    print("Example 4: Disabling Automatic Rate Limiting")
    print("=" * 60)
    
    try:
        # Disable automatic rate limiting
        model = get_model(
            "google:gemini-1.5-flash",
            auto_rate_limiting=False,  # Disable auto rate limiting
        )
        
        print(f"Model created without automatic rate limiting")
        print("Note: Only do this if you have a paid tier with higher limits!")
    except ImportError as e:
        print(f"Note: {e}")
        print("This example requires 'langchain-google-genai' package.")
        print()
        print("Code example:")
        print("  model = get_model(")
        print("      'google:gemini-1.5-flash',")
        print("      auto_rate_limiting=False,")
        print("  )")
    print()


def example_5_environment_variables():
    """
    Example 5: Using environment variables for rate limiting.
    
    You can configure rate limiting via environment variables:
    
    # .env file:
    ENABLE_RATE_LIMITING=true
    RATE_LIMIT_REQUESTS_PER_SECOND=1.0
    GOOGLE_RATE_LIMIT_RPS=2.0
    """
    print("=" * 60)
    print("Example 5: Environment Variable Configuration")
    print("=" * 60)
    
    print("Available environment variables:")
    print("  ENABLE_RATE_LIMITING=true              # Enable rate limiting globally")
    print("  RATE_LIMIT_REQUESTS_PER_SECOND=1.0     # Default RPS for all providers")
    print("  GOOGLE_RATE_LIMIT_RPS=2.0              # Provider-specific RPS")
    print("  DISABLE_RATE_LIMITING_GOOGLE=true      # Disable for specific provider")
    print()
    print("When ENABLE_RATE_LIMITING is set, rate limiting is applied to")
    print("all providers automatically without code changes.")
    print()


def example_6_multiple_requests():
    """
    Example 6: Making multiple requests with rate limiting.
    
    This example shows how rate limiting automatically throttles requests.
    """
    print("=" * 60)
    print("Example 6: Multiple Requests with Rate Limiting")
    print("=" * 60)
    
    # Create a very slow rate limiter for demonstration
    # 0.2 RPS = 1 request every 5 seconds
    limiter = get_gemini_rate_limiter(requests_per_second=0.2)
    
    try:
        model = get_model(
            "google:gemini-1.5-flash",
            rate_limiter=limiter,
        )
        
        print("Created model with 0.2 RPS (1 request every 5 seconds)")
        print("In a real scenario, requests would be throttled automatically.")
        print()
        print("Sample code:")
        print("  messages = [HumanMessage(content='Hello')]")
        print("  for i in range(5):")
        print("      response = model.invoke(messages)  # Auto-throttled")
        print("      print(response.content)")
    except ImportError as e:
        print(f"Note: {e}")
        print("This example requires 'langchain-google-genai' package.")
        print()
        print("The rate limiter is created and ready to use:")
        print(f"  Rate limiter: {limiter}")
        print("  Configured for 0.2 RPS (1 request every 5 seconds)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("Rate Limiting Examples for Google Gemini Models")
    print("*" * 60)
    print("\n")
    
    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable is not set.")
        print("These examples will show the code but cannot make actual API calls.")
        print("Set GOOGLE_API_KEY in your .env file to run actual examples.")
        print("\n")
    
    example_1_basic_gemini_with_rate_limiting()
    example_2_explicit_rate_limiter()
    example_3_using_llm_config()
    example_4_disable_auto_rate_limiting()
    example_5_environment_variables()
    example_6_multiple_requests()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Gemini models automatically get rate limiting (1 RPS)")
    print("  2. Use get_gemini_rate_limiter() for custom rates")
    print("  3. Set ENABLE_RATE_LIMITING=true for global rate limiting")
    print("  4. Use auto_rate_limiting=False to disable automatic limiting")
    print()


if __name__ == "__main__":
    main()
