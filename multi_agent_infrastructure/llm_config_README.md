# LLM Configuration Guide

This guide explains how to configure and use different LLM providers with the Multi-Agent Infrastructure.

## Quick Start

1. **Copy the environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to `.env`:**
   ```env
   OPENAI_API_KEY=sk-your-key-here
   # or
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   # or
   MOONSHOT_API_KEY=sk-your-key-here
   # etc.
   ```

3. **Use in your code:**
   ```python
   from multi_agent_infrastructure import get_model
   
   # Auto-detect based on available API keys
   model = get_model()
   
   # Or explicitly specify
   model = get_model("openai:gpt-4o")
   model = get_model("anthropic:claude-3-5-sonnet-latest")
   model = get_model("moonshot:kimi-k2-5")
   ```

## Supported Providers

| Provider | Environment Variable | Default Model | Notes |
|----------|---------------------|---------------|-------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | GPT-4o, GPT-4, o1 series |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-5-sonnet-latest | Claude 3.5 Sonnet, Opus, Haiku |
| Google | `GOOGLE_API_KEY` | gemini-1.5-flash | Gemini 1.5 Flash/Pro |
| Cohere | `COHERE_API_KEY` | command-r | Command R/R+ |
| Groq | `GROQ_API_KEY` | llama-3.3-70b-versatile | Fast inference for open models |
| Moonshot AI | `MOONSHOT_API_KEY` | kimi-k2-5 | Kimi k2.5, k1.5 |
| OpenRouter | `OPENROUTER_API_KEY` | openai/gpt-4o | Access to 200+ models |
| Ollama | None | llama3.1 | Local models, no API key needed |
| Azure | `AZURE_OPENAI_API_KEY` | gpt-4 | Enterprise OpenAI |

## Usage Examples

### Using Model String Identifiers

```python
from multi_agent_infrastructure import get_model

# Provider:model format (recommended)
model = get_model("openai:gpt-4o")
model = get_model("anthropic:claude-3-5-sonnet-latest")
model = get_model("moonshot:kimi-k2-5")
model = get_model("openrouter:anthropic/claude-3.5-sonnet")

# Model aliases (shorthand)
model = get_model("gpt-4o-mini")  # Resolves to openai:gpt-4o-mini
model = get_model("claude-3-opus")  # Resolves to anthropic:claude-3-opus-latest
model = get_model("kimi-k2-5")  # Resolves to moonshot:kimi-k2-5
```

### Using LLMConfig for Advanced Options

```python
from multi_agent_infrastructure import LLMConfig

config = LLMConfig(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,
)
model = config.get_model()
```

### Moonshot AI (Kimi)

```python
# Using string identifier
model = get_model("moonshot:kimi-k2-5")

# Using LLMConfig
config = LLMConfig(
    provider="moonshot",
    model="kimi-k2-5",
    temperature=0.7,
)
model = config.get_model()
```

### OpenRouter

```python
# Using OpenRouter to access Claude
model = get_model("openrouter:anthropic/claude-3.5-sonnet")

# Using OpenRouter to access DeepSeek
model = get_model("openrouter:deepseek/deepseek-chat")

# Using OpenRouter to access Qwen
model = get_model("openrouter:qwen/qwen-2.5-72b-instruct")
```

### Auto-Detection

If you don't specify a model, the system will auto-detect based on available API keys:

```python
# Uses the first available provider
model = get_model()
```

Priority order for auto-detection:
1. OpenAI
2. Anthropic
3. Moonshot
4. Groq
5. OpenRouter
6. Google

## Model Aliases

Use these shortcuts instead of full provider:model strings:

| Alias | Resolves To |
|-------|-------------|
| `gpt-4o` | openai:gpt-4o |
| `gpt-4o-mini` | openai:gpt-4o-mini |
| `claude-3-5-sonnet` | anthropic:claude-3-5-sonnet-latest |
| `claude-3-opus` | anthropic:claude-3-opus-latest |
| `gemini-1.5-flash` | google:gemini-1.5-flash |
| `kimi-k2-5` | moonshot:kimi-k2-5 |
| `llama-3.3-70b` | groq:llama-3.3-70b-versatile |
| `or-gpt-4o` | openrouter:openai/gpt-4o |
| `or-claude-3.5-sonnet` | openrouter:anthropic/claude-3.5-sonnet |

## Checking Provider Status

```python
from multi_agent_infrastructure import get_provider_info, list_available_providers

# List all providers and their configuration status
for provider, info in get_provider_info().items():
    print(f"{provider}: {'✓ configured' if info['has_key'] else '✗ not configured'}")
    print(f"  Default model: {info['default_model']}")
    print(f"  Env variable: {info['env_key']}")

# List just provider names
providers = list_available_providers()
print(providers)  # ['anthropic', 'azure', 'cohere', 'google', 'groq', 'moonshot', 'ollama', 'openai', 'openrouter']
```

## Integration with Agents

```python
from multi_agent_infrastructure import (
    get_model,
    ResearchAgent,
    CodeAgent,
    AgentRegistry,
    create_orchestrator,
    OrchestratorConfig,
)

# Get model
model = get_model("moonshot:kimi-k2-5")

# Create agents
research = ResearchAgent(model=model)
code = CodeAgent(model=model)

# Register and orchestrate
registry = AgentRegistry()
registry.register("research", research, description="Research agent")
registry.register("code", code, description="Code agent")

config = OrchestratorConfig(supervisor_model=model)
orchestrator = create_orchestrator(registry, config)
```

## Environment Variables

All API keys are loaded from environment variables. The system automatically loads `.env` file when you import and use `get_model()`.

### Required Environment Variables by Provider

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Moonshot AI
MOONSHOT_API_KEY=sk-...

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...

# Groq
GROQ_API_KEY=gsk-...

# Google
GOOGLE_API_KEY=...

# Cohere
COHERE_API_KEY=...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...openai.azure.com/
OPENAI_API_VERSION=2024-02-15-preview
```

## Troubleshooting

### Missing API Key

```
ValueError: No model specified and no API keys found in environment.
```

**Solution:** Add an API key to your `.env` file.

### Missing Package

```
ImportError: langchain-openai
```

**Solution:** Install the required package:
```bash
pip install langchain-openai
```

Or install all provider packages:
```bash
pip install -r requirements.txt
```

### Invalid Model String

```
ValueError: Cannot resolve model 'unknown-model'.
```

**Solution:** Use the `provider:model` format or a known alias.

## Advanced Configuration

### Custom Base URL

For custom endpoints or proxies:

```python
config = LLMConfig(
    provider="openai",
    model="gpt-4o",
    base_url="https://custom-api.example.com/v1",
    api_key="your-api-key",
)
model = config.get_model()
```

### Provider-Specific Parameters

```python
config = LLMConfig(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,
    timeout=60.0,
    max_retries=3,
    additional_kwargs={
        "seed": 42,
        "frequency_penalty": 0.5,
    },
)
model = config.get_model()
```
