"""Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration.

This package implements the Vibe AIGC paradigm from the paper:
"Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration"

Architecture (Paper Section 5):
- MetaPlanner: Decomposes Vibes into workflows (Section 5.2)
- KnowledgeBase: Domain-specific expert knowledge (Section 5.3)
- ToolRegistry: Atomic tool library for content generation (Section 5.4)
"""

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .planner import MetaPlanner
from .llm import LLMClient, LLMConfig
from .executor import WorkflowExecutor, ExecutionStatus, ExecutionResult

# Paper Section 5.3: Domain-Specific Expert Knowledge Base
from .knowledge import (
    KnowledgeBase,
    DomainKnowledge,
    create_knowledge_base
)

# Paper Section 5.4: Atomic Tool Library
from .tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ToolSpec,
    ToolCategory,
    LLMTool,
    TemplateTool,
    CombineTool,
    create_default_registry
)

__version__ = "0.1.2"
__all__ = [
    # Core models
    "Vibe", "WorkflowPlan", "WorkflowNode", "WorkflowNodeType",
    # MetaPlanner (Section 5.2)
    "MetaPlanner", "LLMClient", "LLMConfig",
    # Executor
    "WorkflowExecutor", "ExecutionStatus", "ExecutionResult",
    # Knowledge Base (Section 5.3)
    "KnowledgeBase", "DomainKnowledge", "create_knowledge_base",
    # Tool Registry (Section 5.4)
    "ToolRegistry", "BaseTool", "ToolResult", "ToolSpec", "ToolCategory",
    "LLMTool", "TemplateTool", "CombineTool", "create_default_registry"
]