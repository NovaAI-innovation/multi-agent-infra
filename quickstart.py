"""
Quick Start Example for the Multi-Agent Infrastructure

This is the simplest way to get started with the multi-agent orchestrator.

Setup:
    1. Copy .env.example to .env
    2. Add your API key(s) to .env
    3. Install dependencies: pip install -r requirements.txt
    
Usage:
    python quickstart.py

Supported Providers:
    - OpenAI (GPT-4o, GPT-4o-mini, etc.)
    - Anthropic (Claude 3.5 Sonnet, etc.)
    - Moonshot AI (Kimi k2.5, etc.)
    - OpenRouter (200+ models)
    - Groq, Google, Cohere, Ollama
"""

from __future__ import annotations

import uuid
from langchain_core.messages import HumanMessage

# Import the multi-agent infrastructure
from multi_agent_infrastructure import (
    AgentRegistry,
    create_orchestrator,
    OrchestratorConfig,
    ResearchAgent,
    CodeAgent,
    GeneralAgent,
    # Import the new LLM configuration
    get_model,
    list_available_providers,
    get_provider_info,
)
from multi_agent_infrastructure.core.state import create_initial_state


def setup_model():
    """
    Set up the language model using the model-agnostic approach.
    
    The system automatically:
    1. Loads API keys from .env file
    2. Detects available providers
    3. Creates the appropriate model instance
    
    Returns:
        Configured language model
    """
    print("\n[Step 1] Setting up language model...")
    
    # Show available providers and their status
    print("\n  Available providers:")
    for provider, info in get_provider_info().items():
        status = "+ configured" if info["has_key"] else "- not configured"
        print(f"    - {provider}: {status}")
    
    try:
        # Option 1: Let the system auto-detect based on available API keys
        # model = get_model()
        
        # Option 2: Explicitly specify a model using provider:model format
        # model = get_model("openai:gpt-4o-mini")
        # model = get_model("anthropic:claude-3-5-haiku-latest")
        # model = get_model("moonshot:kimi-k2-5")
        # model = get_model("openrouter:anthropic/claude-3.5-sonnet")
        
        # Option 3: Use model aliases (shorthand)
        # model = get_model("gpt-4o-mini")  # Resolves to openai:gpt-4o-mini
        # model = get_model("claude-3-5-sonnet")  # Resolves to anthropic:claude-3-5-sonnet
        # model = get_model("kimi-k2-5")  # Resolves to moonshot:kimi-k2-5
        
        # Auto-detect based on available API keys
        model = get_model()
        
        # Get model info for display
        model_info = getattr(model, 'model_name', getattr(model, 'model', 'unknown'))
        print(f"  ✓ Using model: {model_info}")
        
        return model
        
    except ValueError as e:
        print(f"\n  ✗ Error: {e}")
        print("\n  To fix this:")
        print("  1. Copy .env.example to .env")
        print("  2. Add at least one API key to .env")
        print("  3. Run this script again")
        print("\n  Supported providers and their environment variables:")
        for provider, info in get_provider_info().items():
            print(f"    - {provider}: {info['env_key']}")
        return None
    except ImportError as e:
        print(f"\n  ✗ Missing package: {e}")
        print("\n  Install required packages:")
        print("    pip install -r requirements.txt")
        return None


def create_agents(registry: AgentRegistry, model) -> None:
    """
    Create and register specialist agents.
    
    Args:
        registry: Agent registry
        model: Language model from get_model()
    """
    print("\n[Step 2] Creating agents...")
    
    # Research Agent
    research = ResearchAgent(
        model=model,
        tools=[],  # Add search tools here (e.g., Tavily, SerpAPI)
    )
    registry.register(
        name="research",
        agent=research,
        description="Research agent for information gathering",
        capabilities=["research", "search", "facts"],
    )
    print("  ✓ Registered: research")
    
    # Code Agent
    code = CodeAgent(
        model=model,
        tools=[],  # Add code execution tools here
    )
    registry.register(
        name="code",
        agent=code,
        description="Code agent for programming tasks",
        capabilities=["coding", "debugging", "python"],
    )
    print("  ✓ Registered: code")
    
    # General Agent (fallback)
    general = GeneralAgent(model=model)
    registry.register(
        name="general",
        agent=general,
        description="General purpose agent",
        capabilities=["general", "conversation"],
    )
    print("  ✓ Registered: general")
    
    print(f"\n  Total agents: {len(registry.list_agents())}")


def run_conversation(orchestrator, session_id: str, user_input: str):
    """
    Run a conversation through the orchestrator.

    Args:
        orchestrator: The compiled orchestrator
        session_id: Unique session identifier
        user_input: User's message
    """
    # Create initial state
    state = create_initial_state(
        session_id=session_id,
        max_iterations=5,
    )
    state["messages"] = [HumanMessage(content=user_input)]

    print("\n" + "=" * 70)
    print("Orchestrator is processing...")
    print("=" * 70 + "\n")

    try:
        # Run the orchestrator with proper configuration for checkpointing
        config = {"configurable": {"thread_id": session_id}}
        result = orchestrator.invoke(state, config=config)

        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        # Show routing history
        print("\n[Routing History]")
        for i, decision in enumerate(result.get("routing_history", []), 1):
            print(f"  {i}. {decision['target_agent']}")
            print(f"     Reason: {decision['reason']}")
            print(f"     Confidence: {decision['confidence']:.2f}")

        # Show final messages
        print("\n[Conversation]")
        for msg in result.get("messages", []):
            msg_type = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', str(msg))

            if msg_type == 'human':
                print(f"\n  You: {content}")
            elif msg_type == 'ai':
                # Truncate long responses
                display_content = content[:500] + "..." if len(content) > 500 else content
                agent_name = getattr(msg, 'name', 'Agent')
                print(f"\n  {agent_name}: {display_content}")

        print("\n" + "=" * 70)
        print("Session completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Quick start example."""
    
    print("=" * 70)
    print("Multi-Agent Infrastructure - Quick Start")
    print("=" * 70)
    
    # Step 1: Set up the model
    model = setup_model()
    if model is None:
        return
    
    # Step 2: Create and register agents
    registry = AgentRegistry()
    create_agents(registry, model)
    
    # Step 3: Create the orchestrator
    print("\n[Step 3] Creating orchestrator...")
    
    config = OrchestratorConfig(
        supervisor_model=model,  # Use LLM for intelligent routing
        max_iterations=5,
        debug=False,
    )
    
    orchestrator = create_orchestrator(registry, config)
    print("  ✓ Orchestrator ready!")
    
    # Step 4: Run conversation
    print("\n[Step 4] Running conversation...")
    print("-" * 70)
    
    session_id = str(uuid.uuid4())
    
    # Get user input
    user_input = input(
        "\nEnter your message (or press Enter for default example):\n> "
    ).strip()
    
    if not user_input:
        user_input = "What are the benefits of using Python for data science, and can you show me a simple example?"
        print(f"Using default: {user_input}")
    
    run_conversation(orchestrator, session_id, user_input)


def demo_with_specific_model(model_string: str):
    """
    Demo with a specific model.
    
    Args:
        model_string: Model identifier (e.g., "openai:gpt-4o", "moonshot:kimi-k2-5")
    """
    print("=" * 70)
    print(f"Demo with model: {model_string}")
    print("=" * 70)
    
    try:
        model = get_model(model_string)
    except (ValueError, ImportError) as e:
        print(f"Error loading model: {e}")
        return
    
    registry = AgentRegistry()
    create_agents(registry, model)
    
    config = OrchestratorConfig(
        supervisor_model=model,
        max_iterations=3,
    )
    
    orchestrator = create_orchestrator(registry, config)
    
    session_id = str(uuid.uuid4())
    user_input = "Explain what a multi-agent system is and when it's useful."
    
    run_conversation(orchestrator, session_id, user_input)


if __name__ == "__main__":
    main()
    
    # Uncomment to test with specific models:
    # demo_with_specific_model("openai:gpt-4o-mini")
    # demo_with_specific_model("anthropic:claude-3-5-haiku-latest")
    # demo_with_specific_model("moonshot:kimi-k2-5")
