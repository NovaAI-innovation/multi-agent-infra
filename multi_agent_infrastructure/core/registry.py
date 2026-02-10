"""
Agent Registry for managing agent registration and retrieval.

Provides a centralized registry for all agents in the system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from multi_agent_infrastructure.agents.base_agent import BaseAgent


class AgentRegistry:
    """
    Registry for managing agents in the multi-agent system.
    
    Supports both compiled StateGraph agents and callable agent functions.
    """
    
    def __init__(self):
        """Initialize an empty agent registry."""
        self._agents: dict[str, CompiledStateGraph] = {}
        self._agent_info: dict[str, dict] = {}
        self._base_agents: dict[str, BaseAgent] = {}
    
    def register(
        self,
        name: str,
        agent: CompiledStateGraph | BaseAgent,
        description: str = "",
        capabilities: list[str] | None = None,
    ) -> None:
        """
        Register an agent in the registry.
        
        Args:
            name: Unique name for the agent
            agent: The agent to register (CompiledStateGraph or BaseAgent)
            description: Description of the agent's capabilities
            capabilities: List of agent capabilities/keywords
            
        Raises:
            ValueError: If agent name is already registered
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered")
        
        if isinstance(agent, CompiledStateGraph):
            self._agents[name] = agent
        else:
            # It's a BaseAgent, compile it
            self._base_agents[name] = agent
            self._agents[name] = agent.compile()
        
        self._agent_info[name] = {
            "name": name,
            "description": description,
            "capabilities": capabilities or [],
        }
    
    def get(self, name: str) -> CompiledStateGraph | None:
        """
        Get a compiled agent by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            The compiled agent or None if not found
        """
        return self._agents.get(name)
    
    def get_base_agent(self, name: str) -> BaseAgent | None:
        """
        Get a base agent (uncompiled) by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            The base agent or None if not found
        """
        return self._base_agents.get(name)
    
    def get_info(self, name: str) -> dict | None:
        """
        Get information about a registered agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent info dict or None if not found
        """
        return self._agent_info.get(name)
    
    def list_agents(self) -> list[str]:
        """
        List all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def list_agent_info(self) -> list[dict]:
        """
        Get information about all registered agents.
        
        Returns:
            List of agent info dictionaries
        """
        return list(self._agent_info.values())
    
    def has_agent(self, name: str) -> bool:
        """
        Check if an agent is registered.
        
        Args:
            name: Name of the agent
            
        Returns:
            True if agent is registered, False otherwise
        """
        return name in self._agents
    
    def unregister(self, name: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            name: Name of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            del self._agent_info[name]
            self._base_agents.pop(name, None)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents.clear()
        self._agent_info.clear()
        self._base_agents.clear()
    
    def get_agent_descriptions(self) -> str:
        """
        Get a formatted string describing all registered agents.
        
        Returns:
            Formatted string with agent descriptions
        """
        descriptions = []
        for name, info in self._agent_info.items():
            caps = ", ".join(info["capabilities"]) if info["capabilities"] else "general"
            descriptions.append(f"- {name}: {info['description']} (capabilities: {caps})")
        return "\n".join(descriptions)


# Global registry instance
_global_registry: AgentRegistry | None = None


def get_global_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.
    
    Returns:
        Global AgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry to a fresh instance."""
    global _global_registry
    _global_registry = AgentRegistry()
