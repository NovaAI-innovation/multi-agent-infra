"""
State management for the multi-agent infrastructure.

Defines the shared state schema used across the orchestrator and all agents.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict
from datetime import datetime
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RoutingDecision(TypedDict):
    """Represents a routing decision made by the supervisor."""
    
    target_agent: str
    """Name of the target agent to route to."""
    
    reason: str
    """Reason for the routing decision."""
    
    confidence: float
    """Confidence score (0-1) of the routing decision."""
    
    timestamp: str
    """ISO timestamp of when the decision was made."""


class AgentConfig(TypedDict, total=False):
    """Configuration for an agent in the system."""
    
    name: str
    """Unique name of the agent."""
    
    description: str
    """Description of the agent's capabilities."""
    
    system_prompt: Optional[str]
    """System prompt for the agent."""
    
    tools: list[str]
    """List of tool names available to the agent."""
    
    model: Optional[str]
    """Model identifier to use for this agent."""
    
    max_iterations: int
    """Maximum number of iterations for the agent."""


class Task(TypedDict):
    """Represents a task in the task queue."""
    
    id: str
    """Unique task identifier."""
    
    type: str
    """Task type/category."""
    
    content: str
    """Task content/description."""
    
    assigned_agent: Optional[str]
    """Agent assigned to this task."""
    
    status: Literal["pending", "in_progress", "completed", "failed"]
    """Current status of the task."""
    
    priority: int
    """Task priority (1-10, higher is more important)."""
    
    created_at: str
    """ISO timestamp of task creation."""
    
    completed_at: Optional[str]
    """ISO timestamp of task completion."""
    
    result: Optional[dict[str, Any]]
    """Task result data."""


class OrchestratorState(TypedDict):
    """
    Global state for the multi-agent orchestrator.
    
    This state is shared across all nodes in the graph and maintains
    the conversation history, routing decisions, and task queue.
    """
    
    # Message history (with reducer for aggregation)
    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation message history."""
    
    # Current routing state
    current_agent: Optional[str]
    """Name of the currently active agent."""
    
    next_agent: Optional[str]
    """Name of the next agent to route to (set by supervisor)."""
    
    # Task management
    task_queue: list[Task]
    """Queue of pending tasks."""
    
    active_tasks: list[Task]
    """Currently active/in-progress tasks."""
    
    completed_tasks: list[Task]
    """Completed tasks history."""
    
    # Routing history
    routing_history: list[RoutingDecision]
    """History of routing decisions made by the supervisor."""
    
    # Shared context/memory
    shared_context: dict[str, Any]
    """Shared context/memory accessible by all agents."""
    
    # Agent configurations
    agent_configs: dict[str, AgentConfig]
    """Configuration for each registered agent."""
    
    # Output/result
    final_output: Optional[str]
    """Final output from the system."""
    
    # Status
    status: Literal["idle", "processing", "waiting_for_agent", "completed", "error"]
    """Current status of the orchestrator."""
    
    error_message: Optional[str]
    """Error message if status is 'error'."""
    
    # Metadata
    session_id: str
    """Unique session identifier."""
    
    created_at: str
    """ISO timestamp of session creation."""
    
    iteration_count: int
    """Number of iterations processed."""
    
    max_iterations: int
    """Maximum allowed iterations."""


def create_initial_state(
    session_id: str,
    max_iterations: int = 10,
    agent_configs: Optional[dict[str, AgentConfig]] = None,
) -> OrchestratorState:
    """
    Create an initial empty state for a new session.
    
    Args:
        session_id: Unique identifier for this session
        max_iterations: Maximum number of iterations allowed
        agent_configs: Optional agent configurations
        
    Returns:
        Initialized OrchestratorState
    """
    now = datetime.utcnow().isoformat()
    return {
        "messages": [],
        "current_agent": None,
        "next_agent": None,
        "task_queue": [],
        "active_tasks": [],
        "completed_tasks": [],
        "routing_history": [],
        "shared_context": {},
        "agent_configs": agent_configs or {},
        "final_output": None,
        "status": "idle",
        "error_message": None,
        "session_id": session_id,
        "created_at": now,
        "iteration_count": 0,
        "max_iterations": max_iterations,
    }
