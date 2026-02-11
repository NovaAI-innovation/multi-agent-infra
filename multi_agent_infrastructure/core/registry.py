"""
Agent Registry for managing agent registration and retrieval.

Provides a centralized registry for all agents in the system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable
from langgraph.graph.state import CompiledStateGraph

from multi_agent_infrastructure.core.logger import get_logger

if TYPE_CHECKING:
    from multi_agent_infrastructure.agents.base_agent import BaseAgent

logger = get_logger(__name__)


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
        self._logger = get_logger(f"{__name__}.AgentRegistry")
        self._logger.debug("AgentRegistry initialized")
    
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
            error_msg = f"Agent '{name}' is already registered"
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._logger.debug(
            f"Registering agent '{name}': description='{description[:50]}...', "
            f"capabilities={capabilities}"
        )
        
        if isinstance(agent, CompiledStateGraph):
            self._agents[name] = agent
            self._logger.debug(f"Agent '{name}' is a CompiledStateGraph")
        else:
            # It's a BaseAgent, compile it
            self._base_agents[name] = agent
            self._agents[name] = agent.compile()
            self._logger.debug(f"Agent '{name}' compiled from BaseAgent")
        
        self._agent_info[name] = {
            "name": name,
            "description": description,
            "capabilities": capabilities or [],
        }
        
        self._logger.info(
            f"Agent '{name}' registered successfully with "
            f"{len(capabilities or [])} capabilities"
        )
    
    def get(self, name: str) -> CompiledStateGraph | None:
        """
        Get a compiled agent by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            The compiled agent or None if not found
        """
        agent = self._agents.get(name)
        if agent is None:
            self._logger.warning(f"Agent '{name}' not found in registry")
        else:
            self._logger.debug(f"Retrieved agent '{name}' from registry")
        return agent
    
    def get_base_agent(self, name: str) -> BaseAgent | None:
        """
        Get a base agent (uncompiled) by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            The base agent or None if not found
        """
        agent = self._base_agents.get(name)
        if agent is None:
            self._logger.warning(f"Base agent '{name}' not found in registry")
        return agent
    
    def get_info(self, name: str) -> dict | None:
        """
        Get information about a registered agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent info dict or None if not found
        """
        info = self._agent_info.get(name)
        if info is None:
            self._logger.warning(f"No info found for agent '{name}'")
        return info
    
    def list_agents(self) -> list[str]:
        """
        List all registered agent names.
        
        Returns:
            List of agent names
        """
        agents = list(self._agents.keys())
        self._logger.debug(f"Listing {len(agents)} registered agents: {agents}")
        return agents
    
    def list_agent_info(self) -> list[dict]:
        """
        Get information about all registered agents.
        
        Returns:
            List of agent info dictionaries
        """
        info_list = list(self._agent_info.values())
        self._logger.debug(f"Retrieved info for {len(info_list)} agents")
        return info_list
    
    def has_agent(self, name: str) -> bool:
        """
        Check if an agent is registered.
        
        Args:
            name: Name of the agent
            
        Returns:
            True if agent is registered, False otherwise
        """
        exists = name in self._agents
        self._logger.debug(f"Checking if agent '{name}' exists: {exists}")
        return exists
    
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
            self._logger.info(f"Agent '{name}' unregistered successfully")
            return True
        
        self._logger.warning(f"Cannot unregister: Agent '{name}' not found")
        return False
    
    def clear(self) -> None:
        """Clear all registered agents."""
        count = len(self._agents)
        self._agents.clear()
        self._agent_info.clear()
        self._base_agents.clear()
        self._logger.info(f"Registry cleared: removed {count} agents")
    
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
        
        result = "\n".join(descriptions)
        self._logger.debug(f"Generated descriptions for {len(descriptions)} agents")
        return result


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
        logger.debug("Global registry instance created")
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry to a fresh instance."""
    global _global_registry
    _global_registry = AgentRegistry()
    logger.info("Global registry reset to fresh instance")
