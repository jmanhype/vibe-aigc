"""Core data models for Vibe AIGC system."""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request for content generation with optional character consistency."""
    
    # Core generation parameters
    prompt: str = Field(..., description="Primary prompt for generation")
    negative_prompt: str = Field("", description="Negative prompt to avoid")
    width: int = Field(512, description="Output width")
    height: int = Field(512, description="Output height")
    steps: int = Field(20, description="Number of sampling steps")
    cfg: float = Field(7.0, description="Classifier-free guidance scale")
    seed: int = Field(0, description="Random seed (0 for random)")
    
    # Video-specific
    frames: int = Field(24, description="Number of frames for video")
    fps: int = Field(24, description="Frames per second for video")
    
    # Character consistency / reference image support
    reference_image: Optional[str] = Field(
        None, 
        description="Path to character/style reference image for consistency"
    )
    character_strength: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0,
        description="How strongly to apply character reference (0.0-1.0)"
    )
    reference_type: str = Field(
        "character",
        description="Type of reference: 'character' (face/person), 'style', or 'composition'"
    )
    
    # LoRA support for character consistency
    character_lora: Optional[str] = Field(
        None,
        description="Path to character-specific LoRA model"
    )
    character_lora_strength: float = Field(
        0.8,
        ge=0.0,
        le=2.0,
        description="Strength of character LoRA (0.0-2.0)"
    )
    
    # Additional LoRAs
    loras: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of LoRAs: [{'path': str, 'strength': float}]"
    )
    
    # Model selection (optional - uses discovery if not specified)
    model: Optional[str] = Field(None, description="Specific model to use")
    vae: Optional[str] = Field(None, description="Specific VAE to use")
    
    # Output
    output_prefix: str = Field("vibe", description="Filename prefix for output")
    
    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class CharacterProfile(BaseModel):
    """Profile for maintaining character consistency across generations."""
    
    name: str = Field(..., description="Character identifier/name")
    reference_images: List[str] = Field(
        default_factory=list,
        description="Paths to reference images of this character"
    )
    lora_path: Optional[str] = Field(
        None,
        description="Path to trained character LoRA if available"
    )
    lora_strength: float = Field(0.8, description="Default LoRA strength for this character")
    
    # Character description for prompt injection
    description: str = Field("", description="Text description of character appearance")
    trigger_words: List[str] = Field(
        default_factory=list,
        description="Trigger words for character LoRA"
    )
    
    # Generation preferences
    preferred_ip_strength: float = Field(0.8, description="Preferred IP-Adapter strength")
    
    def to_generation_params(self) -> Dict[str, Any]:
        """Convert profile to generation parameters."""
        params = {}
        if self.reference_images:
            params["reference_image"] = self.reference_images[0]
            params["character_strength"] = self.preferred_ip_strength
        if self.lora_path:
            params["character_lora"] = self.lora_path
            params["character_lora_strength"] = self.lora_strength
        return params


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