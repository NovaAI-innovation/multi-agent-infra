"""
Pre-built specialist agents for common use cases.

These agents are ready to use in the multi-agent orchestrator.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from multi_agent_infrastructure.agents.base_agent import SimpleReactAgent


class ResearchAgent(SimpleReactAgent):
    """
    Agent specialized in research and information gathering.
    
    Capabilities:
    - Searching for information
    - Summarizing findings
    - Fact-checking
    - Gathering data from multiple sources
    """
    
    DEFAULT_PROMPT = """You are a Research Agent specialized in finding and analyzing information.

Your responsibilities:
1. Search for accurate, relevant information
2. Provide well-sourced, factual responses
3. Synthesize information from multiple sources when needed
4. Acknowledge uncertainty when information is unclear
5. Focus on delivering comprehensive yet concise findings

Always cite your sources when possible and be thorough in your research."""
    
    def __init__(
        self,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the Research Agent.
        
        Args:
            model: Language model to use
            tools: Research tools (search, browser, etc.)
            system_prompt: Custom system prompt (uses default if not provided)
            checkpointer: Optional checkpointer for persistence
        """
        super().__init__(
            name="research",
            description="Specialized agent for research, information gathering, and fact-finding",
            model=model,
            tools=tools,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            checkpointer=checkpointer,
        )
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities."""
        return [
            "research",
            "information gathering",
            "fact-checking",
            "summarization",
            "web search",
            "data collection",
        ]


class CodeAgent(SimpleReactAgent):
    """
    Agent specialized in coding and software development.
    
    Capabilities:
    - Writing code
    - Debugging
    - Code review
    - Explaining code
    - Generating tests
    """
    
    DEFAULT_PROMPT = """You are a Code Agent specialized in software development.

Your responsibilities:
1. Write clean, efficient, well-documented code
2. Debug and fix errors in existing code
3. Explain code and programming concepts clearly
4. Generate tests and ensure code quality
5. Follow best practices and appropriate design patterns
6. Consider edge cases and error handling

Always provide complete, runnable code examples and explain your approach."""
    
    def __init__(
        self,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the Code Agent.
        
        Args:
            model: Language model to use
            tools: Code tools (execution, file operations, etc.)
            system_prompt: Custom system prompt (uses default if not provided)
            checkpointer: Optional checkpointer for persistence
        """
        super().__init__(
            name="code",
            description="Specialized agent for coding, debugging, and software development",
            model=model,
            tools=tools,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            checkpointer=checkpointer,
        )
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities."""
        return [
            "coding",
            "programming",
            "debugging",
            "code review",
            "software development",
            "testing",
            "python",
            "javascript",
            "typescript",
        ]


class AnalysisAgent(SimpleReactAgent):
    """
    Agent specialized in data analysis and evaluation.
    
    Capabilities:
    - Data analysis
    - Pattern recognition
    - Comparative analysis
    - Statistical analysis
    - Report generation
    """
    
    DEFAULT_PROMPT = """You are an Analysis Agent specialized in evaluating data and information.

Your responsibilities:
1. Analyze data and identify patterns, trends, and insights
2. Compare and contrast different options or approaches
3. Provide structured, evidence-based evaluations
4. Generate clear, actionable reports
5. Identify risks, benefits, and trade-offs
6. Use analytical frameworks when appropriate

Always base your analysis on available data and clearly distinguish between facts and interpretations."""
    
    def __init__(
        self,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the Analysis Agent.
        
        Args:
            model: Language model to use
            tools: Analysis tools (calculators, data processors, etc.)
            system_prompt: Custom system prompt (uses default if not provided)
            checkpointer: Optional checkpointer for persistence
        """
        super().__init__(
            name="analysis",
            description="Specialized agent for data analysis, evaluation, and comparative assessment",
            model=model,
            tools=tools,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            checkpointer=checkpointer,
        )
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities."""
        return [
            "analysis",
            "data analysis",
            "evaluation",
            "comparison",
            "pattern recognition",
            "reporting",
            "assessment",
        ]


class GeneralAgent(SimpleReactAgent):
    """
    General-purpose agent for handling any task.
    
    This agent serves as a fallback when no specialized agent is appropriate.
    """
    
    DEFAULT_PROMPT = """You are a General-Purpose Agent capable of handling a wide variety of tasks.

Your responsibilities:
1. Handle general questions and requests
2. Delegate to specialized agents when appropriate
3. Provide helpful, accurate responses
4. Ask clarifying questions when needed
5. Maintain context across the conversation

Be helpful, clear, and thorough in your responses."""
    
    def __init__(
        self,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the General Agent.
        
        Args:
            model: Language model to use
            tools: General tools
            system_prompt: Custom system prompt (uses default if not provided)
            checkpointer: Optional checkpointer for persistence
        """
        super().__init__(
            name="general",
            description="General-purpose agent for handling any task",
            model=model,
            tools=tools,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            checkpointer=checkpointer,
        )
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities."""
        return [
            "general",
            "conversation",
            "question answering",
            "assistance",
        ]


class CreativeAgent(SimpleReactAgent):
    """
    Agent specialized in creative tasks and content generation.
    
    Capabilities:
    - Writing and editing
    - Brainstorming
    - Creative problem solving
    - Content generation
    """
    
    DEFAULT_PROMPT = """You are a Creative Agent specialized in generating creative content and ideas.

Your responsibilities:
1. Generate original, creative content
2. Brainstorm ideas and solutions
3. Help with writing, editing, and refining text
4. Think outside the box to solve problems
5. Adapt tone and style to the context

Be imaginative, engaging, and original while meeting the user's requirements."""
    
    def __init__(
        self,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the Creative Agent.
        
        Args:
            model: Language model to use
            tools: Creative tools
            system_prompt: Custom system prompt (uses default if not provided)
            checkpointer: Optional checkpointer for persistence
        """
        super().__init__(
            name="creative",
            description="Specialized agent for creative tasks, writing, and content generation",
            model=model,
            tools=tools,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            checkpointer=checkpointer,
        )
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities."""
        return [
            "creative writing",
            "brainstorming",
            "content generation",
            "editing",
            "storytelling",
            "ideation",
        ]


class PlanningAgent(SimpleReactAgent):
    """
    Agent specialized in planning and organizing.
    
    Capabilities:
    - Task planning
    - Project organization
    - Scheduling
    - Strategy development
    """
    
    DEFAULT_PROMPT = """You are a Planning Agent specialized in organizing tasks and developing strategies.

Your responsibilities:
1. Break down complex tasks into manageable steps
2. Create structured plans and timelines
3. Identify dependencies and priorities
4. Develop strategies for achieving goals
5. Organize information and resources effectively

Be systematic, thorough, and practical in your planning approach."""
    
    def __init__(
        self,
        model: BaseChatModel | str,
        tools: Sequence[BaseTool | Callable] | None = None,
        system_prompt: str | None = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the Planning Agent.
        
        Args:
            model: Language model to use
            tools: Planning tools
            system_prompt: Custom system prompt (uses default if not provided)
            checkpointer: Optional checkpointer for persistence
        """
        super().__init__(
            name="planning",
            description="Specialized agent for task planning, organization, and strategy development",
            model=model,
            tools=tools,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            checkpointer=checkpointer,
        )
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities."""
        return [
            "planning",
            "organization",
            "strategy",
            "task breakdown",
            "scheduling",
            "project management",
        ]
