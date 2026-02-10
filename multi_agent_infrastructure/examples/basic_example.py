"""
Basic example of using the multi-agent infrastructure.

This example demonstrates:
1. Creating specialist agents with model-agnostic configuration
2. Registering them in the orchestrator
3. Running a conversation through the orchestrator

Setup:
    1. Copy .env.example to .env
    2. Add your API key(s) to .env
    3. Run: python -m multi_agent_infrastructure.examples.basic_example
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Import the multi-agent infrastructure
from multi_agent_infrastructure import (
    AgentRegistry,
    create_orchestrator,
    OrchestratorConfig,
    ResearchAgent,
    CodeAgent,
    AnalysisAgent,
    GeneralAgent,
    # Model-agnostic configuration
    get_model,
    LLMConfig,
    list_available_providers,
    get_provider_info,
)
from multi_agent_infrastructure.core.state import create_initial_state


# Example tools for demonstration
@tool
def search_web(query: str) -> str:
    """Simulate web search (replace with real implementation like Tavily or SerpAPI)."""
    return f"[Simulated search results for: {query}]\n\nFound relevant information about {query}. This is a placeholder for real search results."


@tool
def execute_python(code: str) -> str:
    """Simulate code execution (replace with real implementation like Docker or E2B)."""
    return f"[Simulated execution of Python code]\n\n```python\n{code}\n```\n\nOutput: Code executed successfully (placeholder)."


@tool
def analyze_data(data: str) -> str:
    """Simulate data analysis (replace with real implementation)."""
    return f"[Simulated data analysis]\n\nAnalyzed data: {data}\n\nKey findings:\n- Finding 1\n- Finding 2\n- Finding 3"


def get_model_with_fallback():
    """
    Get a language model with fallback options.
    
    Tries multiple providers in order of preference.
    """
    # Preferred providers in order
    preferred_models = [
        "openai:gpt-4o-mini",
        "anthropic:claude-3-5-haiku-latest",
        "moonshot:kimi-k2-5",
        "groq:llama-3.3-70b-versatile",
        "google:gemini-1.5-flash",
    ]
    
    # Try each model in order
    for model_string in preferred_models:
        try:
            provider = model_string.split(":")[0]
            if not get_provider_info()[provider]["has_key"]:
                continue
            
            model = get_model(model_string)
            print(f"  ✓ Using {model_string}")
            return model
        except (ValueError, ImportError):
            continue
    
    # Last resort: try auto-detection
    try:
        model = get_model()
        print(f"  ✓ Using auto-detected model")
        return model
    except ValueError:
        return None


def create_agents(registry: AgentRegistry, model: Any) -> None:
    """
    Create and register specialist agents.
    
    Args:
        registry: Agent registry
        model: Language model from get_model()
    """
    if model is None:
        print("Skipping agent creation (no model available)")
        return
    
    # Create Research Agent
    research_agent = ResearchAgent(
        model=model,
        tools=[search_web],
    )
    registry.register(
        name="research",
        agent=research_agent,
        description="Research agent for information gathering and fact-finding",
        capabilities=["research", "search", "information gathering"],
    )
    
    # Create Code Agent
    code_agent = CodeAgent(
        model=model,
        tools=[execute_python],
    )
    registry.register(
        name="code",
        agent=code_agent,
        description="Code agent for programming and software development",
        capabilities=["coding", "debugging", "software development"],
    )
    
    # Create Analysis Agent
    analysis_agent = AnalysisAgent(
        model=model,
        tools=[analyze_data],
    )
    registry.register(
        name="analysis",
        agent=analysis_agent,
        description="Analysis agent for data evaluation and assessment",
        capabilities=["analysis", "data analysis", "evaluation"],
    )
    
    # Create General Agent (fallback)
    general_agent = GeneralAgent(
        model=model,
        tools=[],
    )
    registry.register(
        name="general",
        agent=general_agent,
        description="General-purpose agent for handling any task",
        capabilities=["general", "conversation"],
    )


def run_simple_conversation():
    """Run a simple conversation through the orchestrator."""
    print("\n" + "="*60)
    print("Setting up model...")
    print("="*60)
    
    model = get_model_with_fallback()
    
    if model is None:
        print("\n" + "="*60)
        print("DEMO MODE: This example requires a language model.")
        print("="*60)
        print("\nTo run this example:")
        print("1. Copy .env.example to .env")
        print("2. Add at least one API key to .env")
        print("   - OPENAI_API_KEY for OpenAI models")
        print("   - ANTHROPIC_API_KEY for Claude models")
        print("   - MOONSHOT_API_KEY for Moonshot AI models")
        print("   - GROQ_API_KEY for Groq models")
        print("   - etc.")
        print("3. Run the example again")
        print("\n" + "="*60)
        return
    
    # Create registry and agents
    registry = AgentRegistry()
    create_agents(registry, model)
    
    print("\n" + "="*60)
    print("Multi-Agent Orchestrator Example")
    print("="*60)
    print(f"\nRegistered agents: {registry.list_agents()}")
    print("\nAgent details:")
    for info in registry.list_agent_info():
        print(f"  - {info['name']}: {info['description']}")
    
    # Create orchestrator configuration
    config = OrchestratorConfig(
        supervisor_model=model,  # Use LLM for intelligent routing
        max_iterations=5,
        debug=False,
    )
    
    # Create the orchestrator
    orchestrator = create_orchestrator(registry, config)
    
    print("\n" + "="*60)
    print("Starting conversation")
    print("="*60)
    
    # Create initial state
    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(session_id=session_id)
    
    # Add user message
    user_message = "I need to research the best Python web frameworks and then write a simple example with Flask."
    initial_state["messages"] = [HumanMessage(content=user_message)]
    
    print(f"\nUser: {user_message}\n")
    print("Orchestrator processing...\n")
    
    try:
        # Run the orchestrator
        result = orchestrator.invoke(initial_state)
        
        # Print results
        print("\n" + "-"*60)
        print("Routing History:")
        print("-"*60)
        for decision in result.get("routing_history", []):
            print(f"  → {decision['target_agent']}: {decision['reason']} (confidence: {decision['confidence']:.2f})")
        
        print("\n" + "-"*60)
        print("Final Messages:")
        print("-"*60)
        for msg in result.get("messages", [])[-3:]:  # Last 3 messages
            msg_type = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', str(msg))[:200]
            print(f"\n[{msg_type}]: {content}...")
        
        print("\n" + "="*60)
        print("Session completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def demo_different_models():
    """Demonstrate using different models with the same orchestrator."""
    print("\n" + "="*60)
    print("Demo: Different Models")
    print("="*60)
    
    # Check which providers are available
    providers = get_provider_info()
    available = [p for p, info in providers.items() if info["has_key"]]
    
    if not available:
        print("No API keys configured. Please set up your .env file.")
        return
    
    print(f"\nAvailable providers: {available}")
    
    # Try each available provider
    for provider in available:
        print(f"\n--- Testing with {provider} ---")
        try:
            default_model = providers[provider]["default_model"]
            model = get_model(f"{provider}:{default_model}")
            
            registry = AgentRegistry()
            create_agents(registry, model)
            
            config = OrchestratorConfig(
                supervisor_model=model,
                max_iterations=2,
            )
            
            orchestrator = create_orchestrator(registry, config)
            
            session_id = str(uuid.uuid4())
            initial_state = create_initial_state(session_id=session_id)
            initial_state["messages"] = [HumanMessage(content="Say 'Hello from {provider}!'")]
            
            result = orchestrator.invoke(initial_state)
            print(f"  ✓ {provider} works!")
            
        except Exception as e:
            print(f"  ✗ {provider} failed: {e}")


def demo_with_streaming():
    """Run the orchestrator with streaming output."""
    model = get_model_with_fallback()
    
    if model is None:
        print("Streaming example requires a language model.")
        return
    
    registry = AgentRegistry()
    create_agents(registry, model)
    
    config = OrchestratorConfig(
        supervisor_model=model,
        max_iterations=3,
    )
    
    orchestrator = create_orchestrator(registry, config)
    
    print("\n" + "="*60)
    print("Streaming Example")
    print("="*60)
    
    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(session_id=session_id)
    initial_state["messages"] = [HumanMessage(content="Calculate 15 * 23 and explain the result.")]
    
    print("\nStreaming updates:\n")
    
    for update in orchestrator.stream(initial_state, stream_mode="updates"):
        for node, data in update.items():
            print(f"[{node}]: {list(data.keys()) if isinstance(data, dict) else 'data'}")


def main():
    """Main entry point."""
    # Run the basic example
    run_simple_conversation()
    
    # Uncomment to test different models:
    # demo_different_models()
    
    # Uncomment to run streaming example:
    # demo_with_streaming()


if __name__ == "__main__":
    main()
