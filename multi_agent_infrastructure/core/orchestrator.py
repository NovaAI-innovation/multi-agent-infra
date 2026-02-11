"""
Main orchestrator implementation using LangGraph StateGraph.

The orchestrator coordinates multiple agents using a hierarchical supervisor pattern.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Literal, Optional
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
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
from multi_agent_infrastructure.core.logger import (
    get_logger,
    log_state_change,
    log_agent_execution,
    log_routing_decision,
    log_error,
    log_session_start,
    log_session_end,
)

logger = get_logger(__name__)


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
    
    log_level: int = logging.INFO
    """Logging level for the orchestrator."""
    
    session_start_time: Optional[float] = field(default=None, repr=False)
    """Internal: Session start timestamp."""


def create_entry_node() -> Callable[[OrchestratorState], dict]:
    """Create the entry node that initializes the session."""
    def entry_node(state: OrchestratorState) -> dict:
        """Initialize session and set status to processing."""
        session_id = state.get("session_id", "unknown")
        iteration = state.get("iteration_count", 0) + 1
        
        logger.debug(
            f"[Session:{session_id}][Iter:{iteration}][Entry] "
            f"Entering entry node, initializing processing"
        )
        
        result = {
            "status": "processing",
            "iteration_count": iteration,
        }
        
        log_state_change(
            logger=logger,
            session_id=session_id,
            node="entry",
            state_changes={"status": "processing", "iteration_count": iteration},
            iteration=iteration,
        )
        
        return result
    
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
        session_id = state.get("session_id", "unknown")
        iteration = state.get("iteration_count", 0)
        
        logger.info(
            f"[Session:{session_id}][Iter:{iteration}][Agent:{agent_name}] "
            f"Starting agent execution"
        )
        
        # Get messages from state
        messages = state.get("messages", [])
        logger.debug(
            f"[Session:{session_id}][Iter:{iteration}][Agent:{agent_name}] "
            f"Processing {len(messages)} messages"
        )
        
        # Track execution time
        start_time = time.time()
        
        try:
            # Invoke the agent
            logger.debug(
                f"[Session:{session_id}][Iter:{iteration}][Agent:{agent_name}] "
                f"Invoking agent..."
            )
            result = agent.invoke({"messages": messages})
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract new messages from agent result
            new_messages = result.get("messages", [])
            
            logger.info(
                f"[Session:{session_id}][Iter:{iteration}][Agent:{agent_name}] "
                f"Agent execution completed in {duration_ms:.2f}ms, "
                f"produced {len(new_messages)} messages"
            )
            
            # Log detailed agent execution
            log_agent_execution(
                logger=logger,
                session_id=session_id,
                agent_name=agent_name,
                input_messages=messages,
                output_messages=new_messages,
                duration_ms=duration_ms,
                iteration=iteration,
            )
            
            return {
                "messages": new_messages,
                "current_agent": agent_name,
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_error(
                logger=logger,
                session_id=session_id,
                node=f"agent:{agent_name}",
                error=e,
                context={"message_count": len(messages)},
                iteration=iteration,
            )
            raise
    
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
        session_id = state.get("session_id", "unknown")
        iteration = state.get("iteration_count", 0)
        next_agent = state.get("next_agent")
        
        logger.debug(
            f"[Session:{session_id}][Iter:{iteration}][Router] "
            f"Routing decision: next_agent={next_agent!r}"
        )
        
        if next_agent is None:
            logger.debug(
                f"[Session:{session_id}][Iter:{iteration}][Router] "
                f"No next_agent set, routing to supervisor"
            )
            return "supervisor"

        if next_agent == "FINISH":
            logger.info(
                f"[Session:{session_id}][Iter:{iteration}][Router] "
                f"Routing to END (FINISH signal)"
            )
            return "FINISH"

        if registry.has_agent(next_agent):
            logger.info(
                f"[Session:{session_id}][Iter:{iteration}][Router] "
                f"Routing to agent: {next_agent}"
            )
            return next_agent

        # Unknown agent, log and finish
        logger.warning(
            f"[Session:{session_id}][Iter:{iteration}][Router] "
            f"Unknown agent '{next_agent}', attempting fallback"
        )
        
        if registry.list_agents():
            # Default to first available agent
            fallback_agent = registry.list_agents()[0]
            logger.warning(
                f"[Session:{session_id}][Iter:{iteration}][Router] "
                f"Falling back to first available agent: {fallback_agent}"
            )
            return fallback_agent

        logger.error(
            f"[Session:{session_id}][Iter:{iteration}][Router] "
            f"No agents available, routing to END"
        )
        return "FINISH"

    return router


def create_orchestrator(
    registry: AgentRegistry,
    config: Optional[OrchestratorConfig] = None,
) -> Any:  # Return type is the compiled graph
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
    
    # Setup logging level based on config
    if config.debug:
        config.log_level = logging.DEBUG
    
    logger.debug(f"Creating orchestrator with config: {config}")
    logger.info(f"Registering {len(registry.list_agents())} agents: {registry.list_agents()}")

    # Create the state graph
    builder = StateGraph(OrchestratorState)

    # Add entry node
    logger.debug("Adding entry node")
    builder.add_node("entry", create_entry_node())

    # Add supervisor node
    logger.debug("Creating supervisor node")
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
        logger.debug(f"Added agent node: {agent_name}")

    # Add edges
    logger.debug("Adding graph edges")
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "supervisor")

    # Add conditional edges from supervisor
    router = create_router_function(registry)
    # Define path_map for the conditional edges
    path_map = {"FINISH": END}
    for name in agent_names:
        path_map[name] = name
    path_map["supervisor"] = "supervisor"
    
    builder.add_conditional_edges(
        "supervisor",
        router,
        path_map
    )
    logger.debug(f"Added conditional edges from supervisor with paths: {list(path_map.keys())}")

    # Create conditional routing function for agents
    def agent_router(state: OrchestratorState) -> str:
        """Route from agent back to supervisor or END based on next_agent."""
        session_id = state.get("session_id", "unknown")
        iteration = state.get("iteration_count", 0)
        next_agent = state.get("next_agent")
        current_agent = state.get("current_agent", "unknown")

        if next_agent == "FINISH":
            logger.info(
                f"[Session:{session_id}][Iter:{iteration}][AgentRouter:{current_agent}] "
                f"Agent signaled FINISH, routing to END"
            )
            return "__end__"

        # If next_agent is None or a valid agent, return to supervisor
        if next_agent is None or registry.has_agent(next_agent):
            target = "supervisor"
            if next_agent:
                target = f"supervisor (will route to {next_agent})"
            logger.debug(
                f"[Session:{session_id}][Iter:{iteration}][AgentRouter:{current_agent}] "
                f"Routing back to {target}"
            )
            return "supervisor"

        # Default to supervisor if next_agent is invalid
        logger.warning(
            f"[Session:{session_id}][Iter:{iteration}][AgentRouter:{current_agent}] "
            f"Invalid next_agent '{next_agent}', routing to supervisor"
        )
        return "supervisor"

    # Add conditional edges from agents to either supervisor or END
    for agent_name in agent_names:
        builder.add_conditional_edges(
            agent_name,
            agent_router,
            {"supervisor": "supervisor", "__end__": END}
        )
        logger.debug(f"Added conditional edges from agent: {agent_name}")

    # Compile with checkpointing if enabled
    compile_kwargs: dict[str, Any] = {"debug": config.debug}

    if config.enable_checkpointing:
        checkpointer = config.checkpointer or InMemorySaver()
        compile_kwargs["checkpointer"] = checkpointer
        logger.debug("Checkpointing enabled")

    if config.interrupt_before:
        compile_kwargs["interrupt_before"] = config.interrupt_before
        logger.debug(f"Interrupt before nodes: {config.interrupt_before}")

    if config.interrupt_after:
        compile_kwargs["interrupt_after"] = config.interrupt_after
        logger.debug(f"Interrupt after nodes: {config.interrupt_after}")

    logger.info("Compiling orchestrator graph...")
    compiled = builder.compile(**compile_kwargs)
    logger.info("Orchestrator compiled successfully")

    return compiled


def create_simple_orchestrator(
    agents: dict[str, Any],
    supervisor_model: Optional[BaseChatModel] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    debug: bool = False,
) -> Any:  # Return type is the compiled graph
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

    logger.info(f"Creating simple orchestrator with {len(agents)} agents")

    # Create registry and register agents
    registry = AgentRegistry()
    for name, agent in agents.items():
        registry.register(
            name=name,
            agent=agent,
            description=f"Agent: {name}",
        )
        logger.debug(f"Registered agent: {name}")

    # Create config
    config = OrchestratorConfig(
        supervisor_model=supervisor_model,
        checkpointer=checkpointer,
        debug=debug,
    )

    return create_orchestrator(registry, config)
