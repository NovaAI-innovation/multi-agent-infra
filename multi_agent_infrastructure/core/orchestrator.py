"""
Main orchestrator implementation using LangGraph StateGraph.

The orchestrator coordinates multiple agents using a hierarchical supervisor pattern.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from multi_agent_infrastructure.core.state import (
    OrchestratorState,
    create_initial_state,
    Task,
)
from multi_agent_infrastructure.core.registry import AgentRegistry
from multi_agent_infrastructure.core.supervisor import create_supervisor_node
from multi_agent_infrastructure.agents.base_agent import create_agent_wrapper


@dataclass
class OrchestratorConfig:
    """Configuration for the multi-agent orchestrator."""
    
    name: str = "multi_agent_orchestrator"
    """Name of the orchestrator."""
    
    max_iterations: int = 10
    """Maximum number of iterations before forcing completion."""
    
    enable_checkpointing: bool = True
    """Whether to enable state checkpointing for persistence."""
    
    checkpointer: Optional[BaseCheckpointSaver] = None
    """Custom checkpointer instance (uses InMemorySaver if None)."""
    
    debug: bool = False
    """Enable debug mode for verbose logging."""
    
    supervisor_model: Optional[BaseChatModel] = None
    """LLM model for intelligent supervisor routing."""
    
    custom_router: Optional[Callable[[OrchestratorState], str]] = None
    """Custom routing function for supervisor."""
    
    enable_human_in_the_loop: bool = False
    """Enable human approval checkpoints."""
    
    interrupt_before: list[str] = field(default_factory=list)
    """Nodes to interrupt before execution."""
    
    interrupt_after: list[str] = field(default_factory=list)
    """Nodes to interrupt after execution."""


def create_entry_node() -> Callable[[OrchestratorState], dict]:
    """Create the entry node that initializes the session."""
    def entry_node(state: OrchestratorState) -> dict:
        """Initialize session and set status to processing."""
        return {
            "status": "processing",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    return entry_node


def create_agent_node(agent_name: str, agent: Any) -> Callable[[OrchestratorState], dict]:
    """
    Create a node that wraps an agent execution.
    
    Args:
        agent_name: Name of the agent
        agent: The compiled agent graph
        
    Returns:
        Node function for the graph
    """
    def agent_node(state: OrchestratorState) -> dict:
        """Execute the agent with current state."""
        # Get messages from state
        messages = state.get("messages", [])
        
        # Invoke the agent
        result = agent.invoke({"messages": messages})
        
        # Extract new messages from agent result
        new_messages = result.get("messages", [])
        
        return {
            "messages": new_messages,
            "current_agent": agent_name,
        }
    
    return agent_node


def create_router_function(registry: AgentRegistry) -> Callable[[OrchestratorState], str]:
    """
    Create the router function for conditional edges.
    
    Args:
        registry: Agent registry
        
    Returns:
        Router function
    """
    def router(state: OrchestratorState) -> str:
        """Route to the appropriate agent or END."""
        next_agent = state.get("next_agent")
        
        if next_agent is None:
            return "supervisor"
        
        if next_agent == "FINISH":
            return END
        
        if registry.has_agent(next_agent):
            return next_agent
        
        # Unknown agent, log and finish
        if registry.list_agents():
            # Default to first available agent
            return registry.list_agents()[0]
        
        return END
    
    return router


def create_orchestrator(
    registry: AgentRegistry,
    config: Optional[OrchestratorConfig] = None,
) -> StateGraph:
    """
    Create a multi-agent orchestrator using the hierarchical supervisor pattern.
    
    Architecture:
    ```
         ┌─────────┐
         │  START  │
         └────┬────┘
              │
         ┌────▼────┐
         │  Entry  │
         └────┬────┘
              │
         ┌────▼─────────┐
         │  Supervisor  │ (makes routing decisions)
         └────┬─────────┘
              │
        ┌─────┴─────┬─────────┬─────────┐
        │           │         │         │
    ┌───▼───┐  ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
    │Agent 1│  │Agent2│ │Agent3│ │ FINISH│
    └───┬───┘  └──┬───┘ └──┬───┘ └───────┘
        └─────────┴─────────┘
                  │
             ┌────▼────┐
             │   END   │
             └─────────┘
    ```
    
    Args:
        registry: Agent registry with all available agents
        config: Orchestrator configuration
        
    Returns:
        Compiled StateGraph ready for execution
    """
    config = config or OrchestratorConfig()
    
    # Create the state graph
    builder = StateGraph(OrchestratorState)
    
    # Add entry node
    builder.add_node("entry", create_entry_node())
    
    # Add supervisor node
    supervisor = create_supervisor_node(
        registry=registry,
        config=config,
        model=config.supervisor_model,
        custom_router=config.custom_router,
    )
    builder.add_node("supervisor", supervisor)
    
    # Add agent nodes
    agent_names = registry.list_agents()
    for agent_name in agent_names:
        agent = registry.get(agent_name)
        agent_node = create_agent_node(agent_name, agent)
        builder.add_node(agent_name, agent_node)
    
    # Build routing map
    # All agents route back to supervisor
    routing_map = {name: name for name in agent_names}
    routing_map["supervisor"] = "supervisor"
    routing_map[END] = END
    
    # Add edges
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "supervisor")
    
    # Add conditional edges from supervisor
    router = create_router_function(registry)
    builder.add_conditional_edges(
        "supervisor",
        router,
        path_map=list(routing_map.values()) + [END],
    )
    
    # All agents route back to supervisor
    for agent_name in agent_names:
        builder.add_edge(agent_name, "supervisor")
    
    # Compile with checkpointing if enabled
    compile_kwargs: dict[str, Any] = {"debug": config.debug}
    
    if config.enable_checkpointing:
        checkpointer = config.checkpointer or InMemorySaver()
        compile_kwargs["checkpointer"] = checkpointer
    
    if config.interrupt_before:
        compile_kwargs["interrupt_before"] = config.interrupt_before
    
    if config.interrupt_after:
        compile_kwargs["interrupt_after"] = config.interrupt_after
    
    return builder.compile(**compile_kwargs)


def create_simple_orchestrator(
    agents: dict[str, Any],
    supervisor_model: Optional[BaseChatModel] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    debug: bool = False,
) -> StateGraph:
    """
    Create a simple orchestrator with minimal configuration.
    
    This is a convenience function for quickly setting up a multi-agent system.
    
    Args:
        agents: Dictionary mapping agent names to compiled agent graphs
        supervisor_model: Optional LLM for intelligent routing
        checkpointer: Optional checkpointer for persistence
        debug: Enable debug mode
        
    Returns:
        Compiled orchestrator graph
        
    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
        
        # Create agents
        research_agent = create_react_agent(ChatOpenAI(), [search_tool])
        code_agent = create_react_agent(ChatOpenAI(), [execute_code_tool])
        
        # Create orchestrator
        orchestrator = create_simple_orchestrator(
            agents={
                "research": research_agent,
                "code": code_agent,
            },
            supervisor_model=ChatOpenAI(),
        )
        
        # Run
        result = orchestrator.invoke({
            "messages": [("user", "Research Python best practices")],
            "session_id": "session_1",
        })
        ```
    """
    from multi_agent_infrastructure.core.registry import AgentRegistry
    
    # Create registry and register agents
    registry = AgentRegistry()
    for name, agent in agents.items():
        registry.register(
            name=name,
            agent=agent,
            description=f"Agent: {name}",
        )
    
    # Create config
    config = OrchestratorConfig(
        supervisor_model=supervisor_model,
        checkpointer=checkpointer,
        debug=debug,
    )
    
    return create_orchestrator(registry, config)
