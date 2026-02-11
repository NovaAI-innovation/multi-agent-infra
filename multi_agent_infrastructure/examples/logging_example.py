"""
Example: Using Logging in Multi-Agent Infrastructure

This example demonstrates how to use the comprehensive logging system
to monitor and debug multi-agent orchestration.

Usage:
    python -m multi_agent_infrastructure.examples.logging_example
"""

from __future__ import annotations

import logging
import os
import sys

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agent_infrastructure import (
    setup_logging,
    get_logger,
    AgentRegistry,
    create_orchestrator,
    OrchestratorConfig,
    ResearchAgent,
    CodeAgent,
    GeneralAgent,
    get_model,
)
from multi_agent_infrastructure.core.logger import (
    log_agent_execution,
    log_routing_decision,
    log_state_change,
    log_session_start,
    log_session_end,
)


def example_1_basic_logging():
    """Example 1: Setting up basic logging."""
    print("=" * 60)
    print("Example 1: Basic Logging Setup")
    print("=" * 60)
    
    # Setup logging with default settings (INFO level, console output)
    setup_logging(level=logging.INFO)
    
    # Get a logger for your module
    logger = get_logger("my_module")
    
    # Log messages at different levels
    logger.debug("This is a debug message (won't show with INFO level)")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\nLogging setup complete. Check console output above.")
    print()


def example_2_logging_with_file():
    """Example 2: Logging to both console and file."""
    print("=" * 60)
    print("Example 2: Logging to Console and File")
    print("=" * 60)
    
    # Setup logging with file output
    log_file = "orchestrator.log"
    setup_logging(
        level=logging.DEBUG,
        log_file=log_file,
    )
    
    logger = get_logger("file_example")
    logger.info("This message goes to both console and file")
    logger.debug("Debug messages also go to the file")
    
    print(f"\nLog file created: {log_file}")
    print("Check the file for detailed logs.")
    print()


def example_3_custom_log_format():
    """Example 3: Using custom log format."""
    print("=" * 60)
    print("Example 3: Custom Log Format")
    print("=" * 60)
    
    # Custom format string
    custom_format = "%(asctime)s | %(name)s | %(message)s"
    
    setup_logging(
        level=logging.INFO,
        format_string=custom_format,
    )
    
    logger = get_logger("custom_format")
    logger.info("This uses a custom format")
    
    print("\nCustom format applied.")
    print()


def example_4_logging_utilities():
    """Example 4: Using logging utility functions."""
    print("=" * 60)
    print("Example 4: Logging Utility Functions")
    print("=" * 60)
    
    setup_logging(level=logging.DEBUG)
    logger = get_logger("utilities_example")
    
    session_id = "session_123"
    iteration = 1
    
    # Log session start
    log_session_start(
        logger=logger,
        session_id=session_id,
        config={"max_iterations": 10, "debug": True},
    )
    
    # Log state change
    log_state_change(
        logger=logger,
        session_id=session_id,
        node="entry",
        state_changes={"status": "processing", "iteration_count": 1},
        iteration=iteration,
    )
    
    # Log routing decision
    log_routing_decision(
        logger=logger,
        session_id=session_id,
        from_agent=None,
        to_agent="research",
        reason="User asked a research question",
        confidence=0.95,
        method="llm",
        iteration=iteration,
    )
    
    # Log agent execution
    from langchain_core.messages import HumanMessage, AIMessage
    
    log_agent_execution(
        logger=logger,
        session_id=session_id,
        agent_name="research",
        input_messages=[HumanMessage(content="What is Python?")],
        output_messages=[AIMessage(content="Python is a programming language.")],
        duration_ms=1250.5,
        iteration=iteration,
    )
    
    # Log session end
    log_session_end(
        logger=logger,
        session_id=session_id,
        status="completed",
        iterations=3,
        duration_seconds=5.67,
        final_output="Task completed successfully",
    )
    
    print("\nLogging utilities demonstrated.")
    print()


def example_5_orchestrator_with_logging():
    """Example 5: Creating an orchestrator with logging enabled."""
    print("=" * 60)
    print("Example 5: Orchestrator with Logging")
    print("=" * 60)
    
    # Setup logging first
    setup_logging(level=logging.DEBUG)
    
    logger = get_logger("orchestrator_example")
    logger.info("Setting up orchestrator with logging...")
    
    # Note: This example shows the setup code but won't run without API keys
    print("\nCode example (requires API keys to run):")
    print("""
    # Setup logging
    setup_logging(level=logging.DEBUG, log_file="orchestrator.log")
    
    # Create agents
    registry = AgentRegistry()
    model = get_model("gpt-4o-mini")
    
    research_agent = ResearchAgent(model=model)
    code_agent = CodeAgent(model=model)
    general_agent = GeneralAgent(model=model)
    
    registry.register("research", research_agent, "Research specialist")
    registry.register("code", code_agent, "Code specialist")
    registry.register("general", general_agent, "General purpose")
    
    # Create orchestrator with debug enabled
    config = OrchestratorConfig(
        debug=True,
        supervisor_model=model,
        max_iterations=5,
    )
    
    orchestrator = create_orchestrator(registry, config)
    
    # Run - all actions will be logged
    result = orchestrator.invoke({
        "messages": [("user", "Research Python web frameworks")],
        "session_id": "test_session",
    })
    """)
    
    print("\nWith debug=True, you'll see detailed logs including:")
    print("  - State changes at each node")
    print("  - Routing decisions with confidence scores")
    print("  - Agent execution times and I/O")
    print("  - Error details with stack traces")
    print()


def example_6_different_log_levels():
    """Example 6: Understanding different log levels."""
    print("=" * 60)
    print("Example 6: Log Levels Explained")
    print("=" * 60)
    
    print("""
The multi-agent infrastructure uses these log levels:

DEBUG (10):
  - State changes
  - Node entry/exit
  - Internal routing decisions
  - Agent compilation details

AGENT_INPUT (15):
  - Messages sent to agents
  - Tool inputs

AGENT_OUTPUT (16):
  - Messages from agents
  - Tool outputs
  - Execution times

ROUTING (17):
  - Supervisor routing decisions
  - Confidence scores
  - Reasons for routing

INFO (20):
  - Session start/end
  - Agent registration
  - High-level orchestration events

WARNING (30):
  - Fallback routing decisions
  - Missing agents
  - Deprecated features

ERROR (40):
  - Agent execution failures
  - Routing errors
  - Configuration issues

CRITICAL (50):
  - System failures
  - Unrecoverable errors

Use setup_logging(level=logging.DEBUG) to see everything,
or setup_logging(level=logging.INFO) for less verbose output.
""")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("Logging Examples for Multi-Agent Infrastructure")
    print("*" * 60)
    print("\n")
    
    example_1_basic_logging()
    example_2_logging_with_file()
    example_3_custom_log_format()
    example_4_logging_utilities()
    example_5_orchestrator_with_logging()
    example_6_different_log_levels()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Use setup_logging() to configure logging")
    print("  2. Use get_logger(__name__) to get a logger")
    print("  3. Set debug=True in OrchestratorConfig for verbose logs")
    print("  4. Log to file for persistent logs")
    print("  5. Use utility functions for structured logging")
    print()


if __name__ == "__main__":
    main()
