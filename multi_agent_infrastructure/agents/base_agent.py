"""
Base agent class and utilities for creating agents.

Provides a standardized interface for creating LangGraph-based agents
that can be used in the multi-agent orchestrator.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict, Annotated

from multi_agent_infrastructure.core.logger import (
    get_logger,
    log_agent_execution,
    log_tool_execution,
    log_error,
)

logger = get_logger(__name__)


class AgentState(TypedDict):
    """Base state schema for agents."""
    
    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation messages."""


class BaseAgent(ABC):
    """
    Abstract base class for agents in the multi-agent system.
    
    Agents are responsible for specific tasks and communicate through
    a shared state managed by the orchestrator.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent
            description: Description of agent capabilities
            model: Language model to use (instance or string identifier)
            tools: Tools available to this agent
            system_prompt: System prompt for the agent
            checkpointer: Optional checkpointer for persistence
        """
        self.name = name
        self.description = description
        self.model = model
        self.tools = list(tools) if tools else []
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self._compiled: Any | None = None
        self._logger = get_logger(f"{__name__}.{name}")
        
        self._logger.debug(
            f"Initialized agent '{name}': description='{description[:50]}...', "
            f"tools={[t.name if hasattr(t, 'name') else str(t) for t in self.tools]}"
        )
    
    @abstractmethod
    def build(self) -> StateGraph | CompiledStateGraph:
        """
        Build the agent's StateGraph.
        
        Returns:
            StateGraph or CompiledStateGraph representing the agent
        """
        pass
    
    def compile(self) -> CompiledStateGraph:
        """
        Compile the agent into an executable graph.
        
        Returns:
            CompiledStateGraph
        """
        if self._compiled is None:
            self._logger.debug(f"Compiling agent '{self.name}'...")
            start_time = time.time()
            
            try:
                graph = self.build()
                # Handle both StateGraph (needs compilation) and CompiledStateGraph (already compiled)
                if isinstance(graph, CompiledStateGraph):
                    # Already compiled (e.g., from create_react_agent)
                    self._compiled = graph
                    self._logger.debug(f"Agent '{self.name}' was already compiled")
                else:
                    compile_kwargs: dict[str, Any] = {}
                    if self.checkpointer:
                        compile_kwargs["checkpointer"] = self.checkpointer
                        self._logger.debug(f"Using custom checkpointer for '{self.name}'")
                    self._compiled = graph.compile(**compile_kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                self._logger.info(f"Agent '{self.name}' compiled successfully in {duration_ms:.2f}ms")
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self._logger.error(f"Failed to compile agent '{self.name}' after {duration_ms:.2f}ms: {e}")
                raise
                
        return self._compiled
    
    def get_capabilities(self) -> list[str]:
        """
        Get a list of agent capabilities.
        
        Returns:
            List of capability strings
        """
        return []


def create_agent_wrapper(
    name: str,
    model: BaseChatModel | str,
    tools: Sequence[BaseTool | Callable] | None = None,
    system_prompt: str | None = None,
) -> Any:
    """
    Create a simple wrapped agent using LangGraph's create_react_agent.
    
    This is a convenience function for quickly creating agents that can
    be used in the orchestrator.
    
    Args:
        name: Agent name
        model: Language model
        tools: Available tools
        system_prompt: System prompt
        
    Returns:
        Compiled agent graph
    """
    agent_logger = get_logger(f"{__name__}.wrapper.{name}")
    agent_logger.debug(
        f"Creating wrapped agent '{name}' with {len(tools) if tools else 0} tools"
    )
    
    return create_react_agent(
        model=model,
        tools=list(tools) if tools else [],
        prompt=system_prompt,
        name=name,
    )


class SimpleReactAgent(BaseAgent):
    """
    Simple ReAct agent implementation.
    
    Uses LangGraph's create_react_agent for standard tool-calling behavior.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        response_format: dict | type | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the simple ReAct agent.
        
        Args:
            name: Agent name
            description: Agent description
            model: Language model
            tools: Available tools
            system_prompt: System prompt
            response_format: Optional structured output format
            checkpointer: Optional checkpointer
        """
        super().__init__(
            name=name,
            description=description,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )
        self.response_format = response_format
        self._logger.debug(
            f"SimpleReactAgent '{name}' initialized with "
            f"response_format={response_format is not None}"
        )
    
    def build(self) -> CompiledStateGraph:
        """Build using create_react_agent."""
        self._logger.debug(f"Building SimpleReactAgent '{self.name}'...")
        
        # create_react_agent returns a CompiledStateGraph
        return create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.system_prompt,
            name=self.name,
            response_format=self.response_format,
        )


class CustomAgent(BaseAgent):
    """
    Custom agent with full control over the StateGraph.
    
    Subclass this to create agents with custom node logic and edges.
    """
    
    def build(self) -> StateGraph:
        """
        Build the custom agent graph.
        
        Override this method to define custom nodes and edges.
        
        Returns:
            StateGraph representing the agent
        """
        self._logger.debug(f"Building custom agent '{self.name}'...")
        
        builder = StateGraph(AgentState)
        
        # Add your custom nodes here
        self._add_nodes(builder)
        
        # Add your custom edges here
        self._add_edges(builder)
        
        return builder
    
    def _add_nodes(self, builder: StateGraph) -> None:
        """Override to add custom nodes."""
        self._logger.debug(f"Adding nodes for custom agent '{self.name}'")
        pass
    
    def _add_edges(self, builder: StateGraph) -> None:
        """Override to add custom edges."""
        self._logger.debug(f"Adding edges for custom agent '{self.name}'")
        pass
