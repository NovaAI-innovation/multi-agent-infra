# Multi-Agent Infrastructure

## Project Overview

This is a **Multi-Agent Infrastructure** project built on LangGraph, implementing a hierarchical supervisor-based multi-agent orchestration system. The project enables creation of complex AI workflows where a central supervisor intelligently routes tasks to specialized agents based on context and capabilities.

### Key Features

- **Hierarchical Supervisor Pattern**: Central orchestrator routes tasks to specialist agents
- **Model-Agnostic LLM Support**: Auto-detection and unified interface for 8+ LLM providers (OpenAI, Anthropic, Moonshot, Groq, Google, Cohere, OpenRouter, Ollama)
- **State Management**: Shared state with checkpointing for persistence
- **Agent Registry**: Centralized agent registration and management
- **Intelligent Routing**: LLM-based or rule-based routing decisions
- **Pre-built Agents**: Research, Code, Analysis, Creative, Planning, and General agents

## Project Structure

```
C:\Users\casey\Documents\tmp/
├── multi_agent_infrastructure/    # Main implementation module
│   ├── __init__.py                # Public API exports
│   ├── llm_config.py              # Model-agnostic LLM configuration
│   ├── llm_config_README.md       # LLM configuration documentation
│   ├── README.md                  # Module documentation
│   ├── core/                      # Core orchestration components
│   │   ├── state.py               # OrchestratorState and state management
│   │   ├── registry.py            # AgentRegistry for agent management
│   │   ├── supervisor.py          # Supervisor routing logic
│   │   └── orchestrator.py        # Main orchestrator creation
│   ├── agents/                    # Agent implementations
│   │   ├── base_agent.py          # BaseAgent abstract class
│   │   └── specialist_agents.py   # Pre-built specialist agents
│   ├── tools/                     # Tool definitions
│   │   └── basic_tools.py         # Example utility tools
│   └── examples/                  # Usage examples
│       └── basic_example.py       # Complete working example
├── langgraph/                     # Reference LangGraph framework (external)
│   ├── libs/                      # LangGraph library packages
│   └── examples/                  # Example notebooks and patterns
├── quickstart.py                  # Quick start example script
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── MULTI_AGENT_INFRASTRUCTURE_GUIDE.md  # Implementation guide
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Core Framework | LangGraph >= 0.2.0 |
| LLM Integration | LangChain Core + provider packages |
| Data Validation | Pydantic >= 2.0.0 |
| Environment Config | python-dotenv >= 1.0.0 |
| Testing | pytest >= 7.0.0 |

### Supported LLM Providers

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4, o1 series
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Moonshot AI**: Kimi k2.5, Kimi k1.5
- **Google**: Gemini 1.5 Flash, Gemini 1.5 Pro
- **Groq**: Llama 3.3, Mixtral, Gemma (fast inference)
- **Cohere**: Command R, Command R+
- **OpenRouter**: Access to 200+ models via unified API
- **Ollama**: Local models (Llama, Mistral, etc.)
- **Azure**: Azure OpenAI Service

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add at least one API key:

```env
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Or Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or Moonshot AI
MOONSHOT_API_KEY=sk-your-key-here

# Or any other supported provider (see .env.example for all options)
```

## Build and Run Commands

### Run Quick Start Example

```bash
python quickstart.py
```

### Run Basic Example

```bash
python -m multi_agent_infrastructure.examples.basic_example
```

### Run Tests

```bash
pytest
```

## Code Style Guidelines

### Python Style

- **Formatter**: Follow PEP 8 conventions
- **Type Hints**: Use type hints for all function signatures and class attributes
- **Docstrings**: Use Google-style docstrings with Args/Returns sections
- **Imports**: Group imports in this order:
  1. Standard library imports
  2. Third-party imports (langchain, langgraph)
  3. Local module imports
- **String Quotes**: Use double quotes for docstrings, single quotes acceptable for simple strings

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `OrchestratorState`, `ResearchAgent` |
| Functions | snake_case | `create_orchestrator()`, `get_model()` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_MODELS`, `PROVIDER_ENV_KEYS` |
| Private | _leading_underscore | `_compiled`, `_build_system_prompt()` |

### File Organization

```python
"""
Module docstring explaining purpose.
"""

from __future__ import annotations

# Standard library imports
import json
from typing import Any, Optional
from dataclasses import dataclass

# Third-party imports
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph

# Local imports
from multi_agent_infrastructure.core.state import OrchestratorState


# Constants
DEFAULT_ITERATIONS = 10


class ExampleClass:
    """Class docstring."""
    
    def method(self, param: str) -> dict:
        """
        Method docstring.
        
        Args:
            param: Description of parameter
            
        Returns:
            Description of return value
        """
        pass
```

## Testing Instructions

### Running Tests

The project uses pytest for testing:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_orchestrator.py

# Run with async support
pytest --asyncio-mode=auto
```

### Test Structure

Tests should be organized following the module structure:

```
tests/
├── test_state.py           # Tests for state management
├── test_registry.py        # Tests for agent registry
├── test_supervisor.py      # Tests for supervisor logic
├── test_orchestrator.py    # Tests for main orchestrator
└── test_llm_config.py      # Tests for LLM configuration
```

### Writing Tests

```python
import pytest
from multi_agent_infrastructure import AgentRegistry, ResearchAgent


def test_agent_registry():
    """Test agent registration and retrieval."""
    registry = AgentRegistry()
    agent = ResearchAgent(model="openai:gpt-4o-mini")
    
    registry.register("research", agent, description="Test agent")
    
    assert registry.has_agent("research")
    assert registry.get("research") is not None
```

## Architecture Details

### State Management

The `OrchestratorState` TypedDict defines the shared state schema:

```python
class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Optional[str]
    next_agent: Optional[str]
    task_queue: list[Task]
    routing_history: list[RoutingDecision]
    shared_context: dict[str, Any]
    status: Literal["idle", "processing", "completed", "error"]
    session_id: str
    iteration_count: int
    max_iterations: int
```

### Routing Flow

```
START → Entry → Supervisor → (routes to) → Agent → (returns to) → Supervisor → END
                    ↓                                         ↑
                    └───────────────────(loop)────────────────┘
```

### Agent Types

1. **ResearchAgent**: Information gathering, search, fact-finding
2. **CodeAgent**: Programming, debugging, code review
3. **AnalysisAgent**: Data analysis, evaluation, comparison
4. **CreativeAgent**: Writing, content generation, brainstorming
5. **PlanningAgent**: Task planning, organization, strategy
6. **GeneralAgent**: Fallback for general tasks

### Creating Custom Agents

```python
from multi_agent_infrastructure.agents.base_agent import CustomAgent
from langgraph.graph import StateGraph

class MyCustomAgent(CustomAgent):
    def _add_nodes(self, builder: StateGraph) -> None:
        builder.add_node("my_node", my_node_function)
    
    def _add_edges(self, builder: StateGraph) -> None:
        builder.add_edge("my_node", END)
```

## Usage Examples

### Basic Usage

```python
from multi_agent_infrastructure import (
    AgentRegistry,
    create_orchestrator,
    OrchestratorConfig,
    ResearchAgent,
    CodeAgent,
    get_model,
)
from multi_agent_infrastructure.core.state import create_initial_state
from langchain_core.messages import HumanMessage

# Get model (auto-detects from available API keys)
model = get_model()

# Create registry
registry = AgentRegistry()

# Register agents
research = ResearchAgent(model=model, tools=[search_tool])
registry.register("research", research, description="Research agent")

code = CodeAgent(model=model, tools=[execute_tool])
registry.register("code", code, description="Code agent")

# Create orchestrator
config = OrchestratorConfig(supervisor_model=model)
orchestrator = create_orchestrator(registry, config)

# Run
state = create_initial_state(session_id="session_1")
state["messages"] = [HumanMessage(content="Research Python web frameworks")]
result = orchestrator.invoke(state)
```

### Using Specific Models

```python
# Using provider:model format
model = get_model("openai:gpt-4o-mini")
model = get_model("anthropic:claude-3-5-sonnet-latest")
model = get_model("moonshot:kimi-k2-5")

# Using aliases
model = get_model("gpt-4o-mini")  # Resolves to openai:gpt-4o-mini
model = get_model("kimi-k2-5")    # Resolves to moonshot:kimi-k2-5
```

## Security Considerations

### API Key Management

- **Never commit `.env` files** to version control
- API keys are loaded from environment variables only
- The `.env.example` file shows required variables without actual values
- Use separate API keys for development and production

### Safe Code Execution

- The provided tools use safe evaluation (limited builtins for calculator)
- For production code execution, use sandboxed environments (Docker, E2B)
- Current `execute_python` tool is a placeholder that should be replaced

### Input Validation

- All agent inputs are validated through Pydantic models
- State schema uses TypedDict with proper type annotations
- Router validates agent names before routing

## Common Issues and Troubleshooting

### Missing API Key

```
ValueError: No model specified and no API keys found in environment.
```

**Solution**: Add an API key to your `.env` file.

### Missing Package

```
ImportError: langchain-openai
```

**Solution**: Install the required package:

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

**Solution**: Use the `provider:model` format or a known alias.

## Additional Resources

- **LLM Configuration Guide**: `multi_agent_infrastructure/llm_config_README.md`
- **Implementation Guide**: `MULTI_AGENT_INFRASTRUCTURE_GUIDE.md`
- **Module README**: `multi_agent_infrastructure/README.md`
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
