"""Core data models for Vibe AIGC system."""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class Vibe(BaseModel):
    """High-level representation of user's creative intent and aesthetic preferences."""

    description: str = Field(..., description="Primary description of the desired outcome")
    style: Optional[str] = Field(None, description="Aesthetic style preferences")
    constraints: List[str] = Field(default_factory=list, description="Limitations and requirements")
    domain: Optional[str] = Field(None, description="Domain context (e.g., 'visual', 'text', 'audio')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and parameters")


class WorkflowNodeType(str, Enum):
    """Types of workflow nodes for different operation categories."""

    ANALYZE = "analyze"
    GENERATE = "generate"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    COMPOSITE = "composite"


class WorkflowNode(BaseModel):
    """Individual task node in hierarchical workflow decomposition."""

    id: str = Field(..., description="Unique identifier for this node")
    type: WorkflowNodeType = Field(..., description="Category of operation this node performs")
    description: str = Field(..., description="Human-readable description of the task")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")
    dependencies: List[str] = Field(default_factory=list, description="IDs of nodes that must complete first")
    children: List['WorkflowNode'] = Field(default_factory=list, description="Sub-tasks for hierarchical decomposition")
    estimated_duration: Optional[int] = Field(None, description="Estimated execution time in seconds")


class WorkflowPlan(BaseModel):
    """Complete execution plan generated from a Vibe."""

    id: str = Field(..., description="Unique identifier for this plan")
    source_vibe: Vibe = Field(..., description="Original vibe that generated this plan")
    root_nodes: List[WorkflowNode] = Field(..., description="Top-level workflow nodes")
    estimated_total_duration: Optional[int] = Field(None, description="Total estimated execution time")
    created_at: Optional[str] = Field(None, description="ISO timestamp of plan creation")


# Enable forward references for WorkflowNode.children
WorkflowNode.model_rebuild()