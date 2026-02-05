"""Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration.

This package implements the Vibe AIGC paradigm from the paper:
"Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration"

Architecture (Paper Section 5):
- MetaPlanner: Decomposes Vibes into workflows (Section 5.2)
- KnowledgeBase: Domain-specific expert knowledge (Section 5.3)
- ToolRegistry: Atomic tool library for content generation (Section 5.4)
- Agents: Specialized role-based agents (Section 4 examples)
- AssetBank: Character and style consistency management
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

# Multi-Modal Tools (Image, Video, Audio, Search)
from .tools_multimodal import (
    ImageGenerationTool,
    VideoGenerationTool,
    AudioGenerationTool,
    TTSTool,
    SearchTool,
    ScrapeTool,
    register_multimodal_tools,
    create_full_registry
)

# Paper Section 4: Specialized Agents
from .agents import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentResult,
    AgentRegistry,
    WriterAgent,
    ResearcherAgent,
    EditorAgent,
    DirectorAgent,
    DesignerAgent,
    ScreenwriterAgent,
    ComposerAgent,
    create_default_agents
)

# Asset Bank for Consistency
from .assets import (
    AssetBank,
    Character,
    StyleGuide,
    Artifact,
    create_asset_bank
)

__version__ = "0.2.0"
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
    "LLMTool", "TemplateTool", "CombineTool", "create_default_registry",
    # Multi-Modal Tools
    "ImageGenerationTool", "VideoGenerationTool", "AudioGenerationTool",
    "TTSTool", "SearchTool", "ScrapeTool",
    "register_multimodal_tools", "create_full_registry",
    # Agents (Section 4 examples)
    "BaseAgent", "AgentRole", "AgentContext", "AgentResult", "AgentRegistry",
    "WriterAgent", "ResearcherAgent", "EditorAgent", "DirectorAgent",
    "DesignerAgent", "ScreenwriterAgent", "ComposerAgent",
    "create_default_agents",
    # Asset Bank
    "AssetBank", "Character", "StyleGuide", "Artifact", "create_asset_bank"
]
# ComfyUI backend for actual image generation
from .comfyui import ComfyUIBackend, ComfyUIConfig, ComfyUIImageTool, create_comfyui_registry

# Workflow templates
from .workflows import WorkflowLibrary, WorkflowTemplate, create_workflow_library

# Audio generation
from .audio import MusicGenBackend, RiffusionBackend, ElevenLabsBackend, MusicGenerationTool, TTSTool

# MV Pipeline
from .mv_pipeline import MVPipeline, Shot, Storyboard, create_mv
