"""
Model-agnostic LLM configuration utility with dotenv support.

Supports multiple providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
- Google (Gemini)
- Cohere (Command R, etc.)
- Groq (Mixtral, Llama, etc.)
- Moonshot AI (Kimi k1.5, Kimi 2.5, etc.)
- OpenRouter (access to many models)
- Ollama (local models)
- Azure OpenAI

Usage:
    from multi_agent_infrastructure.llm_config import get_model, LLMConfig
    
    # Using string identifier (recommended)
    model = get_model("openai:gpt-4o")
    model = get_model("anthropic:claude-3-5-sonnet-latest")
    model = get_model("moonshot:kimi-k2-5")
    model = get_model("openrouter:anthropic/claude-3.5-sonnet")
    
    # Using LLMConfig for more control
    config = LLMConfig(provider="openai", model="gpt-4o", temperature=0.7)
    model = config.get_model()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langchain_core.rate_limiters import BaseRateLimiter

# Import rate limiter utilities
from multi_agent_infrastructure.rate_limiter import (
    apply_rate_limiter_to_config,
    create_rate_limiter_from_env,
    get_provider_rate_limiter,
)


# Provider to environment variable mapping
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,  # Ollama typically doesn't need an API key for local use
    "azure": "AZURE_OPENAI_API_KEY",
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-1.5-flash",
    "cohere": "command-r",
    "groq": "llama-3.3-70b-versatile",
    "moonshot": "kimi-k2-5",
    "openrouter": "openai/gpt-4o",
    "ollama": "llama3.1",
    "azure": "gpt-4",
}

# Model aliases for common models
MODEL_ALIASES = {
    # OpenAI
    "gpt-4o": "openai:gpt-4o",
    "gpt-4o-mini": "openai:gpt-4o-mini",
    "gpt-4-turbo": "openai:gpt-4-turbo",
    "gpt-4": "openai:gpt-4",
    "gpt-3.5-turbo": "openai:gpt-3.5-turbo",
    
    # Anthropic
    "claude-3-5-sonnet": "anthropic:claude-3-5-sonnet-latest",
    "claude-3-5-haiku": "anthropic:claude-3-5-haiku-latest",
    "claude-3-opus": "anthropic:claude-3-opus-latest",
    "claude-3-sonnet": "anthropic:claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic:claude-3-haiku-20240307",
    
    # Google
    "gemini-1.5-flash": "google:gemini-1.5-flash",
    "gemini-1.5-pro": "google:gemini-1.5-pro",
    "gemini-1.0-pro": "google:gemini-1.0-pro",
    
    # Moonshot AI
    "kimi-k2-5": "moonshot:kimi-k2-5",
    "kimi-k1-5": "moonshot:kimi-k1-5",
    "kimi-k1": "moonshot:kimi-k1",
    
    # Groq
    "llama-3.3-70b": "groq:llama-3.3-70b-versatile",
    "llama-3.1-8b": "groq:llama-3.1-8b-instant",
    "mixtral-8x7b": "groq:mixtral-8x7b-32768",
    "gemma-2-9b": "groq:gemma2-9b-it",
    
    # OpenRouter popular models
    "or-claude-3.5-sonnet": "openrouter:anthropic/claude-3.5-sonnet",
    "or-gpt-4o": "openrouter:openai/gpt-4o",
    "or-llama-3.3-70b": "openrouter:meta-llama/llama-3.3-70b-instruct",
    "or-deepseek": "openrouter:deepseek/deepseek-chat",
    "or-qwen-2.5-72b": "openrouter:qwen/qwen-2.5-72b-instruct",
}


def load_dotenv(dotenv_path: str | None = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        dotenv_path: Optional path to .env file. If None, searches in current directory.
        
    Returns:
        True if .env file was found and loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv as _load_dotenv
        import os as _os
        
        # Find .env file if not specified
        if dotenv_path is None:
            # Search in current directory and parent directories
            current = _os.getcwd()
            for _ in range(5):  # Search up to 5 levels up
                env_path = _os.path.join(current, ".env")
                if _os.path.isfile(env_path):
                    dotenv_path = env_path
                    break
                parent = _os.path.dirname(current)
                if parent == current:
                    break
                current = parent
        
        if dotenv_path is None or not _os.path.isfile(dotenv_path):
            return False
            
        return _load_dotenv(dotenv_path=dotenv_path, encoding="utf-8")
    except ImportError:
        # python-dotenv not installed, try to proceed without it
        return False
    except Exception:
        # Ignore errors loading .env (e.g., encoding issues)
        return False


def get_api_key(provider: str) -> str | None:
    """
    Get the API key for a provider from environment variables.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        
    Returns:
        The API key or None if not found
    """
    env_key = PROVIDER_ENV_KEYS.get(provider)
    if env_key is None:
        return None
    return os.environ.get(env_key)


def check_api_key(provider: str) -> bool:
    """
    Check if API key is configured for a provider.
    
    Args:
        provider: The provider name
        
    Returns:
        True if API key exists or provider doesn't require one
    """
    env_key = PROVIDER_ENV_KEYS.get(provider)
    if env_key is None:
        return True
    return os.environ.get(env_key) is not None


def get_missing_api_keys() -> list[str]:
    """
    Get a list of providers that have missing API keys.
    
    Returns:
        List of provider names with missing keys
    """
    missing = []
    for provider, env_key in PROVIDER_ENV_KEYS.items():
        if env_key and not os.environ.get(env_key):
            missing.append(provider)
    return missing


def resolve_model_string(model_string: str) -> tuple[str, str]:
    """
    Resolve a model string or alias to (provider, model) tuple.
    
    Args:
        model_string: Model identifier like "openai:gpt-4o" or "gpt-4o-mini"
        
    Returns:
        Tuple of (provider, model_name)
        
    Raises:
        ValueError: If model string cannot be resolved
    """
    # Check aliases first
    if model_string in MODEL_ALIASES:
        model_string = MODEL_ALIASES[model_string]
    
    # Parse provider:model format
    if ":" in model_string:
        parts = model_string.split(":", 1)
        provider = parts[0].lower()
        model_name = parts[1]
        return provider, model_name
    
    # Try to infer provider from model name patterns
    model_lower = model_string.lower()
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-"):
        return "openai", model_string
    elif model_lower.startswith("claude-"):
        return "anthropic", model_string
    elif model_lower.startswith("gemini-"):
        return "google", model_string
    elif model_lower.startswith("kimi-"):
        return "moonshot", model_string
    elif model_lower.startswith("command-"):
        return "cohere", model_string
    elif model_lower.startswith("llama-") or model_lower.startswith("mixtral-") or model_lower.startswith("gemma-"):
        # Default to groq for these models, but could be others
        return "groq", model_string
    
    raise ValueError(
        f"Cannot resolve model '{model_string}'. "
        f"Use format 'provider:model' (e.g., 'openai:gpt-4o') or a known alias."
    )


@dataclass
class LLMConfig:
    """
    Configuration for an LLM provider.
    
    This class provides a unified interface for configuring different LLM providers.
    
    Example:
        ```python
        # Simple configuration
        config = LLMConfig(provider="openai", model="gpt-4o")
        model = config.get_model()
        
        # Advanced configuration
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-latest",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
        )
        model = config.get_model()
        
        # Moonshot AI configuration
        config = LLMConfig(provider="moonshot", model="kimi-k2-5")
        model = config.get_model()
        
        # OpenRouter configuration
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            temperature=0.5,
        )
        model = config.get_model()
        ```
    """
    
    provider: str
    """The LLM provider (openai, anthropic, google, cohere, groq, moonshot, openrouter, ollama, azure)."""
    
    model: str | None = None
    """The model name. If None, uses provider's default model."""
    
    temperature: float = 0.7
    """Sampling temperature (0-2)."""
    
    max_tokens: int | None = None
    """Maximum number of tokens to generate."""
    
    top_p: float | None = None
    """Nucleus sampling parameter."""
    
    timeout: float | None = None
    """Request timeout in seconds."""
    
    max_retries: int = 2
    """Maximum number of retries for failed requests."""
    
    api_key: str | None = None
    """API key (if None, reads from environment variable)."""
    
    base_url: str | None = None
    """Base URL for API requests (used for custom endpoints or OpenRouter)."""
    
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional provider-specific kwargs."""
    
    rate_limiter: Optional[Any] = None
    """Optional rate limiter for API requests. Highly recommended for Google/Gemini."""
    
    auto_rate_limiting: bool = True
    """Whether to automatically apply provider-specific rate limiting. Default True."""
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        self.provider = self.provider.lower()
        
        # Use default model if not specified
        if self.model is None:
            self.model = DEFAULT_MODELS.get(self.provider, "unknown")
        
        # Validate provider
        valid_providers = set(PROVIDER_ENV_KEYS.keys())
        if self.provider not in valid_providers:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Supported providers: {', '.join(sorted(valid_providers))}"
            )
    
    def get_api_key(self) -> str | None:
        """Get the API key for this configuration."""
        if self.api_key:
            return self.api_key
        return get_api_key(self.provider)
    
    def check_api_key(self) -> bool:
        """Check if API key is available."""
        if self.api_key:
            return True
        return check_api_key(self.provider)
    
    def _get_openai_model(self) -> BaseChatModel:
        """Create OpenAI model."""
        from langchain_openai import ChatOpenAI
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        
        kwargs.update(self.additional_kwargs)
        return ChatOpenAI(**kwargs)
    
    def _get_anthropic_model(self) -> BaseChatModel:
        """Create Anthropic model."""
        from langchain_anthropic import ChatAnthropic
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        
        kwargs.update(self.additional_kwargs)
        return ChatAnthropic(**kwargs)
    
    def _get_google_model(self) -> BaseChatModel:
        """Create Google (Gemini) model with automatic rate limiting."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        
        # Apply rate limiting (automatic for Google/Gemini)
        kwargs = apply_rate_limiter_to_config(
            provider="google",
            kwargs=kwargs,
            rate_limiter=self.rate_limiter,
            auto_apply=self.auto_rate_limiting,
        )
        
        kwargs.update(self.additional_kwargs)
        return ChatGoogleGenerativeAI(**kwargs)
    
    def _get_cohere_model(self) -> BaseChatModel:
        """Create Cohere model."""
        from langchain_cohere import ChatCohere
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.api_key is not None:
            kwargs["cohere_api_key"] = self.api_key
        
        kwargs.update(self.additional_kwargs)
        return ChatCohere(**kwargs)
    
    def _get_groq_model(self) -> BaseChatModel:
        """Create Groq model."""
        from langchain_groq import ChatGroq
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        
        kwargs.update(self.additional_kwargs)
        return ChatGroq(**kwargs)
    
    def _get_moonshot_model(self) -> BaseChatModel:
        """Create Moonshot AI (Kimi) model."""
        try:
            from langchain_moonshot import ChatMoonshot
        except ImportError:
            # Fallback to OpenAI-compatible interface
            from langchain_openai import ChatOpenAI
            
            base_url = self.base_url or "https://api.moonshot.cn/v1"
            api_key = self.get_api_key()
            
            if not api_key:
                raise ValueError(
                    "Moonshot API key not found. Set MOONSHOT_API_KEY environment variable."
                )
            
            kwargs: dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
                "base_url": base_url,
                "api_key": api_key,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            
            kwargs.update(self.additional_kwargs)
            return ChatOpenAI(**kwargs)
        
        # Use native langchain-moonshot if available
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        
        kwargs.update(self.additional_kwargs)
        return ChatMoonshot(**kwargs)
    
    def _get_openrouter_model(self) -> BaseChatModel:
        """Create OpenRouter model (uses OpenAI-compatible interface)."""
        from langchain_openai import ChatOpenAI
        
        base_url = self.base_url or "https://openrouter.ai/api/v1"
        api_key = self.get_api_key()
        
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "base_url": base_url,
            "api_key": api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        
        # OpenRouter-specific headers for rankings
        default_headers = {
            "HTTP-Referer": "https://github.com/multi-agent-infrastructure",
            "X-Title": "Multi-Agent Infrastructure",
        }
        if "default_headers" in self.additional_kwargs:
            default_headers.update(self.additional_kwargs.pop("default_headers"))
        kwargs["default_headers"] = default_headers
        
        kwargs.update(self.additional_kwargs)
        return ChatOpenAI(**kwargs)
    
    def _get_ollama_model(self) -> BaseChatModel:
        """Create Ollama model (using modern langchain-ollama partner package)."""
        from langchain_ollama import ChatOllama
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["num_predict"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        
        kwargs.update(self.additional_kwargs)
        return ChatOllama(**kwargs)
    
    def _get_azure_model(self) -> BaseChatModel:
        """Create Azure OpenAI model."""
        from langchain_openai import AzureChatOpenAI
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["azure_endpoint"] = self.base_url
        
        # Azure-specific environment variables
        if "AZURE_OPENAI_ENDPOINT" in os.environ and not self.base_url:
            kwargs["azure_endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
        if "OPENAI_API_VERSION" in os.environ:
            kwargs["api_version"] = os.environ["OPENAI_API_VERSION"]
        
        kwargs.update(self.additional_kwargs)
        return AzureChatOpenAI(**kwargs)
    
    def get_model(self) -> BaseChatModel:
        """
        Create and return the configured LLM model.
        
        Returns:
            Configured BaseChatModel instance
            
        Raises:
            ValueError: If API key is missing or provider is not supported
            ImportError: If required package is not installed
        """
        provider_method = getattr(self, f"_get_{self.provider}_model", None)
        if provider_method is None:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        return provider_method()


def get_model(
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.7,
    rate_limiter: Optional[Any] = None,
    auto_rate_limiting: bool = True,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Get an LLM model using a model-agnostic approach.
    
    This function follows the LangGraph pattern of accepting a string identifier
    for the model (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet").
    
    Environment variables are loaded from .env file automatically.
    
    Priority for determining model/provider:
    1. Explicit arguments passed to this function
    2. LLM_MODEL and LLM_PROVIDER environment variables
    3. Auto-detection from available API keys
    
    Args:
        model: Model string identifier (e.g., "openai:gpt-4o", "gpt-4o-mini").
               Can also be a known alias like "claude-3-5-sonnet".
        provider: Provider name if not included in model string (e.g., "openai").
        temperature: Sampling temperature (default: 0.7).
        rate_limiter: Optional rate limiter to use. If None, may auto-apply based on provider.
        auto_rate_limiting: Whether to auto-apply provider-specific rate limiting (default: True).
        **kwargs: Additional arguments passed to LLMConfig.
        
    Returns:
        Configured BaseChatModel instance
        
    Raises:
        ValueError: If model string cannot be resolved or API key is missing
        ImportError: If required package is not installed
        
    Example:
        ```python
        # Load environment variables from .env file automatically
        from multi_agent_infrastructure.llm_config import get_model
        
        # Using full model string
        model = get_model("openai:gpt-4o")
        model = get_model("anthropic:claude-3-5-sonnet-latest")
        model = get_model("moonshot:kimi-k2-5")
        model = get_model("openrouter:anthropic/claude-3.5-sonnet")
        
        # Using model aliases (shorter)
        model = get_model("gpt-4o-mini")  # Automatically resolves to openai
        model = get_model("claude-3-opus")  # Automatically resolves to anthropic
        model = get_model("kimi-k2-5")  # Automatically resolves to moonshot
        
        # With custom temperature
        model = get_model("openai:gpt-4o", temperature=0.5)
        
        # With additional parameters
        model = get_model(
            "anthropic:claude-3-5-sonnet-latest",
            temperature=0.7,
            max_tokens=4096,
        )
        
        # With rate limiting (recommended for Google Gemini)
        from multi_agent_infrastructure.rate_limiter import get_gemini_rate_limiter
        limiter = get_gemini_rate_limiter(requests_per_second=1.0)
        model = get_model("google:gemini-1.5-flash", rate_limiter=limiter)
        ```
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # If no provider specified, check environment variable
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER")
    
    # If no model specified, check environment variable first
    if model is None:
        model = os.environ.get("LLM_MODEL")
    
    # If still no model specified, try to build from provider or use defaults
    if model is None:
        if provider:
            # Use default model for the specified provider
            default_model = DEFAULT_MODELS.get(provider)
            if default_model:
                model = f"{provider}:{default_model}"
        else:
            # Auto-detect from available API keys (fallback)
            if os.environ.get("OPENAI_API_KEY"):
                model = "openai:gpt-4o-mini"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                model = "anthropic:claude-3-5-haiku-latest"
            elif os.environ.get("MOONSHOT_API_KEY"):
                model = "moonshot:kimi-k2-5"
            elif os.environ.get("GROQ_API_KEY"):
                model = "groq:llama-3.3-70b-versatile"
            elif os.environ.get("OPENROUTER_API_KEY"):
                model = "openrouter:openai/gpt-4o-mini"
            elif os.environ.get("GOOGLE_API_KEY"):
                model = "google:gemini-1.5-flash"
            else:
                raise ValueError(
                    "No model specified and no API keys found in environment. "
                    "Please set LLM_PROVIDER and LLM_MODEL in your .env file, "
                    "or provide a model explicitly."
                )
    
    # If model has provider prefix (e.g., "openai:gpt-4o"), extract it
    if model and ":" in model:
        provider_name, model_name = model.split(":", 1)
        provider = provider_name  # Update provider from model string
    elif model:
        # Model without provider prefix
        # Check if it's an alias first
        if model in MODEL_ALIASES:
            model = MODEL_ALIASES[model]
            provider_name, model_name = model.split(":", 1)
            provider = provider_name
        elif provider:
            # Use explicitly specified provider
            model = f"{provider}:{model}"
            provider_name, model_name = provider, model
        else:
            # Try to infer provider from model name patterns
            try:
                provider_name, model_name = resolve_model_string(model)
                model = f"{provider_name}:{model_name}"
                provider = provider_name
            except ValueError as e:
                raise ValueError(
                    f"Cannot resolve model '{model}'. "
                    f"Use format 'provider:model' (e.g., 'openai:gpt-4o'), "
                    f"a known alias, or set LLM_PROVIDER in your .env file."
                ) from e
    else:
        raise ValueError("No model specified.")
    
    # Final resolution to ensure we have provider_name and model_name
    if ":" in model:
        provider_name, model_name = model.split(":", 1)
    else:
        raise ValueError(
            f"Invalid model format: '{model}'. "
            f"Use format 'provider:model' (e.g., 'openai:gpt-4o')."
        )
    
    # Create config and get model
    config = LLMConfig(
        provider=provider_name,
        model=model_name,
        temperature=temperature,
        rate_limiter=rate_limiter,
        auto_rate_limiting=auto_rate_limiting,
        **kwargs,
    )
    
    return config.get_model()


def list_available_providers() -> list[str]:
    """Get a list of available provider names."""
    return sorted(PROVIDER_ENV_KEYS.keys())


def list_model_aliases() -> dict[str, str]:
    """Get a dictionary of model aliases and their resolved values."""
    return MODEL_ALIASES.copy()


def get_provider_info() -> dict[str, dict[str, str]]:
    """
    Get information about all supported providers.
    
    Returns:
        Dictionary mapping provider names to their info (env_key, default_model)
    """
    return {
        provider: {
            "env_key": PROVIDER_ENV_KEYS.get(provider) or "None required",
            "default_model": DEFAULT_MODELS.get(provider, "N/A"),
            "has_key": check_api_key(provider),
        }
        for provider in PROVIDER_ENV_KEYS.keys()
    }


# Auto-load environment variables when module is imported
load_dotenv()
