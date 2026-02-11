# Multi-Agent Infrastructure

A production-ready, hierarchical supervisor-based multi-agent system built on LangGraph.

## Architecture

```
┌─────────────────┐
│   Supervisor    │  (Orchestrator that routes to specialists)
│   Orchestrator  │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Research│ │Code  │ │Analysis│ │General│
└───────┘ └──────┘ └──────┘ └──────┘
```

## Features

- **Hierarchical Supervisor Pattern**: Central orchestrator intelligently routes tasks to specialist agents
- **State Management**: Shared state with checkpointing for persistence
- **Agent Registry**: Centralized agent registration and management
- **Intelligent Routing**: LLM-based or rule-based routing decisions
- **Pre-built Agents**: Research, Code, Analysis, Creative, Planning, and General agents
- **Extensible**: Easy to add custom agents and tools

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from multi_agent_infrastructure import (
    AgentRegistry,
    create_orchestrator,
    OrchestratorConfig,
    ResearchAgent,
    CodeAgent,
)
from multi_agent_infrastructure.core.state import create_initial_state
from langchain_core.messages import HumanMessage

# Create model
model = ChatOpenAI(model="gpt-4o-mini")

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

# Provide configuration with thread_id for checkpointing
config = {"configurable": {"thread_id": "session_1"}}
result = orchestrator.invoke(state, config=config)
print(result)
```

## Project Structure

```
multi_agent_infrastructure/
├── __init__.py
├── README.md
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── state.py          # State management (OrchestratorState)
│   ├── registry.py       # Agent registry
│   ├── supervisor.py     # Supervisor/routing logic
│   └── orchestrator.py   # Main orchestrator creation
├── agents/
│   ├── __init__.py
│   ├── base_agent.py     # Base agent class
│   └── specialist_agents.py  # Pre-built agents
├── tools/
│   ├── __init__.py
│   └── basic_tools.py    # Example tools
└── examples/
    ├── __init__.py
    └── basic_example.py  # Usage examples
```

## Components

### State Management

The `OrchestratorState` TypedDict defines the shared state:

```python
class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Optional[str]
    next_agent: Optional[str]
    task_queue: list[Task]
    routing_history: list[RoutingDecision]
    shared_context: dict[str, Any]
    # ... more fields
```

### Agent Registry

Centralized registry for managing agents:

```python
registry = AgentRegistry()
registry.register("research", research_agent, description="Research specialist")
agent = registry.get("research")
```

### Specialist Agents

Pre-built agents for common tasks:

- **ResearchAgent**: Information gathering, search, fact-finding
- **CodeAgent**: Programming, debugging, code review
- **AnalysisAgent**: Data analysis, evaluation, comparison
- **CreativeAgent**: Writing, content generation, brainstorming
- **PlanningAgent**: Task planning, organization, strategy
- **GeneralAgent**: Fallback for general tasks

### Routing

The supervisor can use:
1. **LLM-based routing**: Intelligent decision-making with context
2. **Rule-based routing**: Keyword and pattern matching
3. **Custom router**: User-defined routing function

## Advanced Usage

### Custom Router

```python
def custom_router(state: OrchestratorState) -> str:
    # Custom routing logic
    last_message = state["messages"][-1].content
    if "urgent" in last_message.lower():
        return "priority_agent"
    return "general"

config = OrchestratorConfig(custom_router=custom_router)
```

### Custom Agent

```python
from multi_agent_infrastructure.agents.base_agent import CustomAgent

class MyCustomAgent(CustomAgent):
    def _add_nodes(self, builder):
        builder.add_node("my_node", my_node_function)
    
    def _add_edges(self, builder):
        builder.add_edge("my_node", END)
```

### Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

config = OrchestratorConfig(
    checkpointer=SqliteSaver.from_conn_string(":memory:")
)
```

## Examples

See `examples/basic_example.py` for a complete working example.

Run it:
```bash
export OPENAI_API_KEY=your_key
python -m multi_agent_infrastructure.examples.basic_example
```

## License

MIT
