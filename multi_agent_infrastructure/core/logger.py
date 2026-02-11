"""
Logging utilities for the multi-agent infrastructure.

Provides structured logging for orchestrator, agents, and supervisor actions.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any, Optional
from functools import wraps


# Custom log levels for agent-specific logging
AGENT_INPUT = 15  # Between DEBUG (10) and INFO (20)
AGENT_OUTPUT = 16
ROUTING_DECISION = 17

logging.addLevelName(AGENT_INPUT, "AGENT_INPUT")
logging.addLevelName(AGENT_OUTPUT, "AGENT_OUTPUT")
logging.addLevelName(ROUTING_DECISION, "ROUTING")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    return logger


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration for the multi-agent infrastructure.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string
        log_file: Optional file to log to (in addition to console)
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
        )
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Configure root logger for multi_agent_infrastructure
    root_logger = logging.getLogger("multi_agent_infrastructure")
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_state_change(
    logger: logging.Logger,
    session_id: str,
    node: str,
    state_changes: dict[str, Any],
    iteration: Optional[int] = None,
) -> None:
    """
    Log a state change in the orchestrator.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        node: Node that made the change
        state_changes: Dictionary of state changes
        iteration: Current iteration number
    """
    prefix = f"[Session:{session_id}]"
    if iteration is not None:
        prefix += f"[Iter:{iteration}]"
    prefix += f"[{node}]"
    
    changes_str = ", ".join(f"{k}={v!r}" for k, v in state_changes.items())
    logger.debug(f"{prefix} State change: {changes_str}")


def log_agent_execution(
    logger: logging.Logger,
    session_id: str,
    agent_name: str,
    input_messages: list,
    output_messages: list,
    duration_ms: Optional[float] = None,
    iteration: Optional[int] = None,
) -> None:
    """
    Log agent execution details.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        agent_name: Name of the executing agent
        input_messages: Input messages to the agent
        output_messages: Output messages from the agent
        duration_ms: Execution duration in milliseconds
        iteration: Current iteration
    """
    prefix = f"[Session:{session_id}]"
    if iteration is not None:
        prefix += f"[Iter:{iteration}]"
    prefix += f"[Agent:{agent_name}]"
    
    # Log input (at AGENT_INPUT level)
    input_summary = _summarize_messages(input_messages)
    logger.log(AGENT_INPUT, f"{prefix} INPUT: {input_summary}")
    
    # Log output (at AGENT_OUTPUT level)
    output_summary = _summarize_messages(output_messages)
    duration_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
    logger.log(AGENT_OUTPUT, f"{prefix} OUTPUT{duration_str}: {output_summary}")


def log_routing_decision(
    logger: logging.Logger,
    session_id: str,
    from_agent: Optional[str],
    to_agent: str,
    reason: str,
    confidence: float,
    method: str = "unknown",
    iteration: Optional[int] = None,
) -> None:
    """
    Log a routing decision made by the supervisor.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        from_agent: Current agent (None if starting)
        to_agent: Target agent (or 'FINISH')
        reason: Reason for routing decision
        confidence: Confidence score
        method: Routing method (llm, rule, custom)
        iteration: Current iteration
    """
    prefix = f"[Session:{session_id}]"
    if iteration is not None:
        prefix += f"[Iter:{iteration}]"
    prefix += "[Supervisor]"
    
    from_str = from_agent or "START"
    confidence_pct = confidence * 100
    
    logger.log(
        ROUTING_DECISION,
        f"{prefix} ROUTING: {from_str} -> {to_agent} | "
        f"method={method}, confidence={confidence_pct:.0f}%, reason={reason!r}"
    )


def log_tool_execution(
    logger: logging.Logger,
    session_id: str,
    agent_name: str,
    tool_name: str,
    tool_input: Any,
    tool_output: Any,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    iteration: Optional[int] = None,
) -> None:
    """
    Log tool execution details.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        agent_name: Agent using the tool
        tool_name: Name of the tool
        tool_input: Input to the tool
        tool_output: Output from the tool
        duration_ms: Execution duration
        error: Error message if failed
        iteration: Current iteration
    """
    prefix = f"[Session:{session_id}]"
    if iteration is not None:
        prefix += f"[Iter:{iteration}]"
    prefix += f"[Agent:{agent_name}][Tool:{tool_name}]"
    
    duration_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
    
    if error:
        logger.error(f"{prefix} FAILED{duration_str}: error={error!r}")
    else:
        input_summary = _truncate_str(str(tool_input), 100)
        output_summary = _truncate_str(str(tool_output), 100)
        logger.debug(
            f"{prefix} EXECUTED{duration_str}: input={input_summary!r}, "
            f"output={output_summary!r}"
        )


def log_error(
    logger: logging.Logger,
    session_id: str,
    node: str,
    error: Exception,
    context: Optional[dict] = None,
    iteration: Optional[int] = None,
) -> None:
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        node: Node where error occurred
        error: Exception that was raised
        context: Additional context
        iteration: Current iteration
    """
    prefix = f"[Session:{session_id}]"
    if iteration is not None:
        prefix += f"[Iter:{iteration}]"
    prefix += f"[{node}]"
    
    context_str = ""
    if context:
        context_str = " | Context: " + ", ".join(f"{k}={v!r}" for k, v in context.items())
    
    logger.exception(f"{prefix} ERROR: {type(error).__name__}: {error}{context_str}")


def log_session_start(
    logger: logging.Logger,
    session_id: str,
    config: dict[str, Any],
) -> None:
    """
    Log session start with configuration.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        config: Session configuration
    """
    prefix = f"[Session:{session_id}]"
    config_str = ", ".join(f"{k}={v!r}" for k, v in config.items())
    logger.info(f"{prefix} SESSION START: {config_str}")


def log_session_end(
    logger: logging.Logger,
    session_id: str,
    status: str,
    iterations: int,
    duration_seconds: Optional[float] = None,
    final_output: Optional[str] = None,
) -> None:
    """
    Log session end with summary.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        status: Final status (completed, error, etc.)
        iterations: Number of iterations
        duration_seconds: Total duration
        final_output: Final output summary
    """
    prefix = f"[Session:{session_id}]"
    duration_str = f" duration={duration_seconds:.2f}s" if duration_seconds else ""
    output_summary = ""
    if final_output:
        output_summary = f" output={_truncate_str(final_output, 100)!r}"
    
    logger.info(
        f"{prefix} SESSION END: status={status}, iterations={iterations}"
        f"{duration_str}{output_summary}"
    )


def _summarize_messages(messages: list) -> str:
    """Create a summary of messages for logging."""
    if not messages:
        return "(no messages)"
    
    count = len(messages)
    if count == 1:
        msg = messages[0]
        content = getattr(msg, 'content', str(msg))
        return _truncate_str(content, 150)
    else:
        # Show first and last message
        first = messages[0]
        last = messages[-1]
        first_content = getattr(first, 'content', str(first))
        last_content = getattr(last, 'content', str(last))
        return (
            f"({count} msgs) first={_truncate_str(first_content, 50)!r} ... "
            f"last={_truncate_str(last_content, 50)!r}"
        )


def _truncate_str(s: str, max_length: int) -> str:
    """Truncate string to max_length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def timed_execution(logger: logging.Logger, operation: str):
    """
    Decorator to time and log function execution.
    
    Args:
        logger: Logger instance
        operation: Name of the operation being timed
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.debug(f"{operation} completed in {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(f"{operation} failed after {duration:.2f}ms: {e}")
                raise
        return wrapper
    return decorator
