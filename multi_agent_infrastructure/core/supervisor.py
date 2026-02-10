"""
Supervisor node implementation for the multi-agent orchestrator.

The supervisor analyzes the current state and makes routing decisions
using an LLM or rule-based logic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Literal

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from multi_agent_infrastructure.core.state import (
    OrchestratorState,
    RoutingDecision,
    create_initial_state,
)
from multi_agent_infrastructure.core.registry import AgentRegistry

if TYPE_CHECKING:
    from multi_agent_infrastructure.core.orchestrator import OrchestratorConfig


SUPERVISOR_SYSTEM_PROMPT = """You are an intelligent orchestrator supervisor for a multi-agent system.

Your job is to analyze the current conversation and state, then decide which specialized agent should handle the next step.

## Available Agents
{agent_descriptions}

## Routing Guidelines

1. Analyze the user's request and conversation context
2. Select the most appropriate agent based on:
   - The agent's described capabilities
   - The nature of the current task
   - Previous routing history (avoid unnecessary switching)

3. Available routing targets:
   - Any agent name from the list above
   - "FINISH" - when the task is complete and no more agents are needed

4. Provide a brief reason for your routing decision

## Response Format
Respond with a JSON object in this exact format:
{{
    "target_agent": "agent_name_or_FINISH",
    "reason": "explanation of why this agent was selected",
    "confidence": 0.95
}}

Confidence should be between 0 and 1.
"""


class SupervisorNode:
    """
    Supervisor node that makes routing decisions for the orchestrator.
    
    Can use either an LLM for intelligent routing or rule-based logic.
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        config: OrchestratorConfig,
        model: BaseChatModel | None = None,
        custom_router: Callable[[OrchestratorState], str] | None = None,
    ):
        """
        Initialize the supervisor node.
        
        Args:
            registry: Agent registry containing all available agents
            config: Orchestrator configuration
            model: Optional LLM for intelligent routing
            custom_router: Optional custom routing function
        """
        self.registry = registry
        self.config = config
        self.model = model
        self.custom_router = custom_router
        
        # Build system prompt with agent descriptions
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with current agent descriptions."""
        agent_descriptions = self.registry.get_agent_descriptions()
        if not agent_descriptions:
            agent_descriptions = "- general: General purpose agent for any task"
        return SUPERVISOR_SYSTEM_PROMPT.format(agent_descriptions=agent_descriptions)
    
    def _parse_routing_response(self, content: str) -> tuple[str, str, float]:
        """
        Parse the routing response from the LLM.
        
        Args:
            content: JSON string response from LLM
            
        Returns:
            Tuple of (target_agent, reason, confidence)
        """
        try:
            # Try to parse JSON
            data = json.loads(content)
            target = data.get("target_agent", "FINISH")
            reason = data.get("reason", "No reason provided")
            confidence = float(data.get("confidence", 0.5))
            return target, reason, confidence
        except json.JSONDecodeError:
            # Fallback: try to extract from text
            content_lower = content.lower()
            if "finish" in content_lower:
                return "FINISH", "Parsed from text: task appears complete", 0.7
            
            # Check for agent names in the response
            for agent_name in self.registry.list_agents():
                if agent_name.lower() in content_lower:
                    return agent_name, f"Parsed from text: mentioned {agent_name}", 0.6
            
            return "FINISH", "Could not parse routing decision, defaulting to FINISH", 0.5
    
    def __call__(self, state: OrchestratorState) -> dict:
        """
        Make a routing decision based on the current state.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            State update with routing decision
        """
        # Check if we should stop
        if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
            return {
                "next_agent": "FINISH",
                "status": "completed",
                "routing_history": [
                    *state.get("routing_history", []),
                    RoutingDecision(
                        target_agent="FINISH",
                        reason="Maximum iterations reached",
                        confidence=1.0,
                        timestamp=__import__("datetime").datetime.utcnow().isoformat(),
                    )
                ],
            }
        
        # Use custom router if provided
        if self.custom_router:
            target = self.custom_router(state)
            return {
                "next_agent": target,
                "routing_history": [
                    *state.get("routing_history", []),
                    RoutingDecision(
                        target_agent=target,
                        reason="Custom router decision",
                        confidence=1.0,
                        timestamp=__import__("datetime").datetime.utcnow().isoformat(),
                    )
                ],
            }
        
        # Use LLM-based routing if model is available
        if self.model:
            return self._llm_route(state)
        
        # Fallback to simple rule-based routing
        return self._rule_based_route(state)
    
    def _llm_route(self, state: OrchestratorState) -> dict:
        """Use LLM to make a routing decision."""
        from datetime import datetime
        
        messages = state.get("messages", [])
        
        # Prepare conversation context
        conversation = []
        for msg in messages[-10:]:  # Last 10 messages for context
            if isinstance(msg, SystemMessage):
                continue
            conversation.append(f"{msg.type}: {msg.content}")
        
        context = "\n".join(conversation)
        current_agent = state.get("current_agent", "None")
        
        user_prompt = f"""Current agent: {current_agent}

Conversation context:
{context}

Based on the conversation, which agent should handle the next step?

Respond with the JSON format specified in your instructions."""
        
        # Call LLM
        response = self.model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        content = response.content if hasattr(response, 'content') else str(response)
        target, reason, confidence = self._parse_routing_response(content)
        
        # Validate target agent exists
        if target != "FINISH" and not self.registry.has_agent(target):
            # Try to find a matching agent or default to first available
            available = self.registry.list_agents()
            if available:
                target = available[0]
                reason = f"Original target not found, defaulting to {target}"
            else:
                target = "FINISH"
                reason = "No agents available, finishing"
        
        routing_decision = RoutingDecision(
            target_agent=target,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        return {
            "next_agent": target,
            "routing_history": [*state.get("routing_history", []), routing_decision],
            "status": "processing" if target != "FINISH" else "completed",
        }
    
    def _rule_based_route(self, state: OrchestratorState) -> dict:
        """Simple rule-based routing as fallback."""
        from datetime import datetime
        
        messages = state.get("messages", [])
        if not messages:
            # No messages, finish
            return {
                "next_agent": "FINISH",
                "routing_history": [
                    RoutingDecision(
                        target_agent="FINISH",
                        reason="No messages to process",
                        confidence=1.0,
                        timestamp=datetime.utcnow().isoformat(),
                    )
                ],
                "status": "completed",
            }
        
        # Get last user message
        last_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                last_message = msg
                break
        
        if not last_message:
            return {
                "next_agent": "FINISH",
                "routing_history": [
                    RoutingDecision(
                        target_agent="FINISH",
                        reason="No user message found",
                        confidence=1.0,
                        timestamp=datetime.utcnow().isoformat(),
                    )
                ],
                "status": "completed",
            }
        
        content = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()
        
        # Simple keyword-based routing
        target = "general"
        reason = "Default routing"
        
        if any(kw in content for kw in ["research", "search", "find", "look up", "information about"]):
            target = "research"
            reason = "Detected research-related keywords"
        elif any(kw in content for kw in ["code", "program", "function", "script", "debug", "error", "python", "javascript"]):
            target = "code"
            reason = "Detected code-related keywords"
        elif any(kw in content for kw in ["analyze", "analysis", "compare", "evaluate", "assess"]):
            target = "analysis"
            reason = "Detected analysis-related keywords"
        
        # Check if target agent exists
        if not self.registry.has_agent(target):
            available = self.registry.list_agents()
            if available:
                target = available[0]
                reason = f"Original target not available, using {target}"
            else:
                target = "FINISH"
                reason = "No agents available"
        
        routing_decision = RoutingDecision(
            target_agent=target,
            reason=reason,
            confidence=0.7,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        return {
            "next_agent": target,
            "routing_history": [*state.get("routing_history", []), routing_decision],
            "status": "processing" if target != "FINISH" else "completed",
        }


def create_supervisor_node(
    registry: AgentRegistry,
    config: OrchestratorConfig,
    model: BaseChatModel | None = None,
    custom_router: Callable[[OrchestratorState], str] | None = None,
) -> SupervisorNode:
    """
    Create a supervisor node for the orchestrator.
    
    Args:
        registry: Agent registry
        config: Orchestrator configuration
        model: Optional LLM for intelligent routing
        custom_router: Optional custom routing function
        
    Returns:
        Configured SupervisorNode instance
    """
    return SupervisorNode(registry, config, model, custom_router)
