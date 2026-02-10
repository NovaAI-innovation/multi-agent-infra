"""
Multi-Agent Infrastructure based on LangGraph.

A hierarchical supervisor-based multi-agent system for building complex agent workflows.
"""

from multi_agent_infrastructure.core.orchestrator import create_orchestrator, OrchestratorConfig
from multi_agent_infrastructure.core.registry import AgentRegistry
from multi_agent_infrastructure.core.state import OrchestratorState, AgentConfig
from multi_agent_infrastructure.agents.base_agent import BaseAgent
from multi_agent_infrastructure.agents.specialist_agents import (
    ResearchAgent,
    CodeAgent,
    AnalysisAgent,
    GeneralAgent,
)
from multi_agent_infrastructure.llm_config import (
    get_model,
    LLMConfig,
    load_dotenv,
    list_available_providers,
    list_model_aliases,
    get_provider_info,
)

__all__ = [
    # Core components
    "create_orchestrator",
    "OrchestratorConfig",
    "AgentRegistry",
    "OrchestratorState",
    "AgentConfig",
    # Agents
    "BaseAgent",
    "ResearchAgent",
    "CodeAgent",
    "AnalysisAgent",
    "GeneralAgent",
    # LLM Configuration
    "get_model",
    "LLMConfig",
    "load_dotenv",
    "list_available_providers",
    "list_model_aliases",
    "get_provider_info",
]

__version__ = "0.1.0"
