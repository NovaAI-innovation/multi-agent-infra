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
from multi_agent_infrastructure.rate_limiter import (
    RateLimiterConfig,
    create_rate_limiter,
    create_rate_limiter_from_env,
    get_provider_rate_limiter,
    get_gemini_rate_limiter,
)
from multi_agent_infrastructure.core.logger import (
    get_logger,
    setup_logging,
    log_agent_execution,
    log_routing_decision,
    log_state_change,
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
    # Rate Limiting
    "RateLimiterConfig",
    "create_rate_limiter",
    "create_rate_limiter_from_env",
    "get_provider_rate_limiter",
    "get_gemini_rate_limiter",
    # Logging
    "get_logger",
    "setup_logging",
    "log_agent_execution",
    "log_routing_decision",
    "log_state_change",
]

__version__ = "0.1.0"
