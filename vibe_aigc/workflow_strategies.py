"""Workflow Generation Strategies — Pattern-Based Architecture.

Design Patterns Applied:
========================

1. STRATEGY PATTERN
   - VideoGenerationStrategy interface defines the contract
   - Concrete strategies (LTX2Strategy, WanStrategy, AnimateDiffStrategy) implement it
   - Each knows its required nodes, model types, and workflow structure
   
2. FACTORY PATTERN  
   - StrategyFactory.create() selects the best strategy based on:
     * Available models (discovered from ComfyUI)
     * Hardware constraints (VRAM)
     * User preferences (quality vs speed)
   
3. TEMPLATE METHOD PATTERN
   - WorkflowTemplate defines the skeleton: load → encode → sample → decode → save
   - Subclasses override hooks for model-specific nodes
   
4. BUILDER PATTERN
   - ComfyWorkflowBuilder constructs arbitrary workflows node-by-node
   - Used internally by strategies for flexibility

Trade-offs:
===========
- Strategy: Clean separation but requires implementing each backend
- Factory: Centralized selection but factory can grow complex
- Template: Great for similar structures but rigid for outliers
- Builder: Maximum flexibility but verbose for simple cases

The combination gives us: discoverable, extensible, and maintainable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type
from enum import Enum
import random


# =============================================================================
# CORE TYPES
# =============================================================================

class Capability(Enum):
    """What a model can do."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    UPSCALE = "upscale"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class ModelSpec:
    """Discovered model specification."""
    filename: str
    category: str  # checkpoints, unet, vae, clip, etc.
    capabilities: List[Capability]
    vram_required: float
    quality_tier: int  # 1-10
    loader_nodes: List[str]  # Which ComfyUI nodes can load this
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        return self.filename.replace(".safetensors", "").replace("_", " ").title()


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    success: bool
    outputs: List[str] = field(default_factory=list)  # URLs or paths
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class VideoRequest:
    """Request for video generation."""
    prompt: str
    negative_prompt: str = "blurry, static, low quality, watermark"
    frames: int = 25
    width: int = 512
    height: int = 320
    fps: int = 24
    steps: int = 20
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)


# =============================================================================
# BUILDER PATTERN — Flexible workflow construction
# =============================================================================

class ComfyWorkflowBuilder:
    """Builds ComfyUI workflow JSON node-by-node.
    
    Usage:
        workflow = (ComfyWorkflowBuilder()
            .add_node("1", "CheckpointLoaderSimple", ckpt_name="model.safetensors")
            .add_node("2", "CLIPTextEncode", text="prompt", clip=("1", 1))
            .build())
    """
    
    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
    
    def add_node(
        self, 
        node_id: str, 
        class_type: str, 
        **inputs
    ) -> "ComfyWorkflowBuilder":
        """Add a node to the workflow.
        
        Inputs can be:
        - Direct values: text="hello", steps=20
        - Node references: clip=("1", 0) means node "1" output 0
        """
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Node reference: (node_id, output_index)
                processed_inputs[key] = [str(value[0]), value[1]]
            else:
                processed_inputs[key] = value
        
        self._nodes[node_id] = {
            "class_type": class_type,
            "inputs": processed_inputs
        }
        return self
    
    def next_id(self) -> str:
        """Get next auto-incrementing node ID."""
        self._counter += 1
        return str(self._counter)
    
    def build(self) -> Dict[str, Any]:
        """Return the complete workflow JSON."""
        return self._nodes.copy()


# =============================================================================
# STRATEGY PATTERN — Different video generation approaches
# =============================================================================

class VideoGenerationStrategy(ABC):
    """Abstract strategy for video generation.
    
    Each concrete strategy knows:
    - What models it needs (model_type, text_encoder_type, vae_type)
    - How to build the ComfyUI workflow
    - Node-specific parameters
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        pass
    
    @property
    @abstractmethod
    def required_capabilities(self) -> List[Capability]:
        """What capabilities the models must have."""
        pass
    
    @abstractmethod
    def can_use_models(self, available_models: Dict[str, List[ModelSpec]]) -> bool:
        """Check if this strategy can work with available models."""
        pass
    
    @abstractmethod
    def select_models(
        self, 
        available_models: Dict[str, List[ModelSpec]],
        max_vram: float
    ) -> Dict[str, ModelSpec]:
        """Select the best models for this strategy."""
        pass
    
    @abstractmethod
    def build_workflow(
        self, 
        request: VideoRequest,
        models: Dict[str, ModelSpec]
    ) -> Dict[str, Any]:
        """Build the ComfyUI workflow JSON."""
        pass
    
    def quality_score(self, models: Dict[str, ModelSpec]) -> float:
        """Estimate quality score for this strategy with given models."""
        if not models:
            return 0.0
        return sum(m.quality_tier for m in models.values()) / len(models)


class LTX2Strategy(VideoGenerationStrategy):
    """Strategy for LTX-2 19B video generation.
    
    LTX-2 is a state-of-the-art video model but requires:
    - Separate UNET/checkpoint loading (not in unet folder, in checkpoints)
    - T5 text encoder (separate from model)
    - Specific VAE
    - LTX-specific conditioning and sampling nodes
    """
    
    @property
    def name(self) -> str:
        return "LTX-2 Video"
    
    @property
    def required_capabilities(self) -> List[Capability]:
        return [Capability.TEXT_TO_VIDEO]
    
    def can_use_models(self, available_models: Dict[str, List[ModelSpec]]) -> bool:
        """Check for LTX-2 checkpoint and required components."""
        checkpoints = available_models.get("checkpoints", [])
        vaes = available_models.get("vae", [])
        
        has_ltx = any("ltx" in m.filename.lower() for m in checkpoints)
        has_vae = any("ltx" in m.filename.lower() for m in vaes)
        
        return has_ltx and has_vae
    
    def select_models(
        self, 
        available_models: Dict[str, List[ModelSpec]],
        max_vram: float
    ) -> Dict[str, ModelSpec]:
        """Select LTX-2 model and components."""
        models = {}
        
        # Find LTX checkpoint
        for m in available_models.get("checkpoints", []):
            if "ltx" in m.filename.lower() and m.vram_required <= max_vram:
                models["checkpoint"] = m
                break
        
        # Find LTX VAE
        for m in available_models.get("vae", []):
            if "ltx" in m.filename.lower():
                models["vae"] = m
                break
        
        # Find T5 text encoder (in clip or text_encoders)
        for category in ["clip", "text_encoders"]:
            for m in available_models.get(category, []):
                if "t5" in m.filename.lower():
                    models["text_encoder"] = m
                    break
            if "text_encoder" in models:
                break
        
        return models
    
    def build_workflow(
        self, 
        request: VideoRequest,
        models: Dict[str, ModelSpec]
    ) -> Dict[str, Any]:
        """Build LTX-2 specific workflow.
        
        CORRECT PATTERN:
        - LTX-2 checkpoint does NOT include CLIP
        - Must use LTXAVTextEncoderLoader for text encoding
        - Use LTXVBaseSampler for proper video sampling
        """
        
        checkpoint = models.get("checkpoint")
        vae = models.get("vae")
        text_encoder = models.get("text_encoder")
        
        if not checkpoint:
            raise ValueError("Missing checkpoint for LTX-2")
        
        builder = ComfyWorkflowBuilder()
        
        # 1. Load checkpoint (MODEL at 0, CLIP at 1 is NULL, VAE at 2)
        builder.add_node("1", "CheckpointLoaderSimple",
            ckpt_name=checkpoint.filename)
        
        # 2. Load text encoder SEPARATELY (LTX-2 needs this!)
        te_file = text_encoder.filename if text_encoder else "t5xxl_fp8_e4m3fn_scaled.safetensors"
        builder.add_node("2", "LTXAVTextEncoderLoader",
            text_encoder=te_file,
            ckpt_name=checkpoint.filename,
            device="default")
        
        # 3. Encode prompts using the separate text encoder
        builder.add_node("3", "CLIPTextEncode",
            text=request.prompt,
            clip=("2", 0))
        
        builder.add_node("4", "CLIPTextEncode",
            text=request.negative_prompt,
            clip=("2", 0))
        
        # 4. LTX conditioning (adds frame rate)
        builder.add_node("5", "LTXVConditioning",
            positive=("3", 0),
            negative=("4", 0),
            frame_rate=float(request.fps))
        
        # 5. Model sampling wrapper
        builder.add_node("6", "ModelSamplingLTXV",
            model=("1", 0),
            max_shift=2.05,
            base_shift=0.95)
        
        # 6. Scheduler (generates sigmas)
        builder.add_node("7", "LTXVScheduler",
            steps=request.steps,
            max_shift=2.05,
            base_shift=0.95,
            stretch=True,
            terminal=0.1)
        
        # 7. Sampling components
        builder.add_node("8", "RandomNoise", noise_seed=request.seed)
        # CFGGuider supports both positive AND negative conditioning
        builder.add_node("9", "CFGGuider", 
            model=("6", 0), 
            positive=("5", 0),  # LTXVConditioning output 0
            negative=("5", 1),  # LTXVConditioning output 1
            cfg=3.0)  # Lower CFG for video
        builder.add_node("10", "KSamplerSelect", sampler_name="euler")
        
        # 8. LTXVBaseSampler - proper video sampler
        builder.add_node("11", "LTXVBaseSampler",
            model=("6", 0),
            vae=("1", 2),
            width=request.width,
            height=request.height,
            num_frames=request.frames,
            guider=("9", 0),
            sampler=("10", 0),
            sigmas=("7", 0),
            noise=("8", 0))
        
        # 9. Decode latent to images
        builder.add_node("12", "VAEDecode",
            samples=("11", 0),
            vae=("1", 2))
        
        # 10. Save as animated webp
        builder.add_node("13", "SaveAnimatedWEBP",
            images=("12", 0),
            filename_prefix="vibe_ltx",
            fps=float(request.fps),
            lossless=False,
            quality=85,
            method="default")
        
        return builder.build()


class WanVideoStrategy(VideoGenerationStrategy):
    """Strategy for Wan 2.x video generation.
    
    Based on working 3090 workflow analysis:
    - DiffusionModelLoaderKJ (KJNodes) for Wan models
    - CLIPLoader with type='wan' for umt5 text encoder
    - VAELoader for wan VAE
    - BasicScheduler + KSamplerSelect for sampling
    """
    
    @property
    def name(self) -> str:
        return "Wan Video"
    
    @property
    def required_capabilities(self) -> List[Capability]:
        return [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO]
    
    def can_use_models(self, available_models: Dict[str, List[ModelSpec]]) -> bool:
        """Check for Wan models in unet/diffusion_models."""
        unets = available_models.get("unet", []) + available_models.get("diffusion_models", [])
        vaes = available_models.get("vae", [])
        
        has_wan = any("wan" in m.filename.lower() for m in unets)
        has_vae = any("wan" in m.filename.lower() for m in vaes)
        
        return has_wan and has_vae
    
    def select_models(
        self, 
        available_models: Dict[str, List[ModelSpec]],
        max_vram: float
    ) -> Dict[str, ModelSpec]:
        """Select Wan model and components."""
        models = {}
        
        # Find Wan UNET (prefer higher quality)
        unets = available_models.get("unet", []) + available_models.get("diffusion_models", [])
        wan_models = [m for m in unets if "wan" in m.filename.lower() and m.vram_required <= max_vram]
        if wan_models:
            # Prefer 2.2 over 2.1, HIGH over LOW, KJ format
            wan_models.sort(key=lambda m: (
                "2.2" in m.filename or "2_2" in m.filename,
                "high" in m.filename.lower(),
                "_KJ" in m.filename,  # KJNodes format preferred
                m.quality_tier
            ), reverse=True)
            models["unet"] = wan_models[0]
        
        # Find Wan VAE
        for m in available_models.get("vae", []):
            if "wan" in m.filename.lower():
                models["vae"] = m
                break
        
        # Find umt5 text encoder (fp8 scaled version)
        for category in ["clip", "text_encoders"]:
            for m in available_models.get(category, []):
                if "umt5" in m.filename.lower() and "fp8" in m.filename.lower():
                    models["text_encoder"] = m
                    break
            if "text_encoder" in models:
                break
        
        return models
    
    def build_workflow(
        self, 
        request: VideoRequest,
        models: Dict[str, ModelSpec]
    ) -> Dict[str, Any]:
        """Build Wan-specific workflow based on working 3090 pattern.
        
        CORRECT PATTERN (from 3090 analysis):
        - DiffusionModelLoaderKJ for model loading (not UNETLoader)
        - CLIPLoader with type='wan' (not CLIPLoaderGGUF)
        - BasicScheduler + KSamplerSelect for sampling
        - SamplerCustomAdvanced for execution
        """
        
        unet = models.get("unet")
        vae = models.get("vae")
        text_encoder = models.get("text_encoder")
        
        if not unet or not vae:
            raise ValueError("Missing required models for Wan Video")
        
        builder = ComfyWorkflowBuilder()
        
        # 1. Load diffusion model using DiffusionModelLoaderKJ (KJNodes)
        builder.add_node("1", "DiffusionModelLoaderKJ",
            model_path=unet.filename,
            weight_dtype="default",
            load_dtype="default",
            fp8_patch=False)
        
        # 2. Load CLIP/text encoder with type='wan'
        te_file = text_encoder.filename if text_encoder else "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        builder.add_node("2", "CLIPLoader",
            clip_name=te_file,
            type="wan",
            device="default")
        
        # 3. Load VAE
        builder.add_node("3", "VAELoader",
            vae_name=vae.filename)
        
        # 4-5. Encode prompts
        builder.add_node("4", "CLIPTextEncode",
            text=request.prompt,
            clip=("2", 0))
        
        builder.add_node("5", "CLIPTextEncode",
            text=request.negative_prompt,
            clip=("2", 0))
        
        # 6. Model sampling (SD3 style for Wan)
        builder.add_node("6", "ModelSamplingSD3",
            shift=5.0,
            model=("1", 0))
        
        # 7. BasicScheduler
        builder.add_node("7", "BasicScheduler",
            scheduler="simple",
            steps=request.steps,
            denoise=1.0,
            model=("6", 0))
        
        # 8. Empty latent for video (using batch for frames)
        builder.add_node("8", "EmptyLatentImage",
            width=request.width,
            height=request.height,
            batch_size=request.frames)
        
        # 9. Random noise
        builder.add_node("9", "RandomNoise",
            noise_seed=request.seed)
        
        # 10. CFG Guider
        builder.add_node("10", "CFGGuider",
            model=("6", 0),
            positive=("4", 0),
            negative=("5", 0),
            cfg=6.0)
        
        # 11. KSamplerSelect
        builder.add_node("11", "KSamplerSelect",
            sampler_name="euler")
        
        # 12. SamplerCustomAdvanced
        builder.add_node("12", "SamplerCustomAdvanced",
            noise=("9", 0),
            guider=("10", 0),
            sampler=("11", 0),
            sigmas=("7", 0),
            latent_image=("8", 0))
        
        # 13. VAE Decode
        builder.add_node("13", "VAEDecode",
            samples=("12", 0),
            vae=("3", 0))
        
        # 14. Save as video (VHS_VideoCombine)
        builder.add_node("14", "VHS_VideoCombine",
            images=("13", 0),
            frame_rate=request.fps,
            loop_count=0,
            filename_prefix="vibe_wan",
            format="video/h264-mp4")
        
        return builder.build()


class AnimateDiffStrategy(VideoGenerationStrategy):
    """Strategy for AnimateDiff video generation.
    
    AnimateDiff works with SD1.5/SDXL base models plus motion modules.
    Simpler setup but lower quality than newer models.
    """
    
    @property
    def name(self) -> str:
        return "AnimateDiff"
    
    @property
    def required_capabilities(self) -> List[Capability]:
        return [Capability.TEXT_TO_VIDEO]
    
    def can_use_models(self, available_models: Dict[str, List[ModelSpec]]) -> bool:
        """Check for SD checkpoint + AnimateDiff motion module."""
        checkpoints = available_models.get("checkpoints", [])
        # AnimateDiff modules are usually in a separate folder
        # For now, assume if we have a checkpoint we can try
        has_sd = any(
            any(x in m.filename.lower() for x in ["sd", "dreamshaper", "realistic"])
            for m in checkpoints
        )
        return has_sd
    
    def select_models(
        self, 
        available_models: Dict[str, List[ModelSpec]],
        max_vram: float
    ) -> Dict[str, ModelSpec]:
        """Select SD checkpoint for AnimateDiff."""
        models = {}
        
        for m in available_models.get("checkpoints", []):
            if m.vram_required <= max_vram:
                # Prefer SD 1.5 for AnimateDiff compatibility
                if "sd" in m.filename.lower() or "v1" in m.filename.lower():
                    models["checkpoint"] = m
                    break
        
        return models
    
    def build_workflow(
        self, 
        request: VideoRequest,
        models: Dict[str, ModelSpec]
    ) -> Dict[str, Any]:
        """Build AnimateDiff workflow."""
        
        checkpoint = models.get("checkpoint")
        if not checkpoint:
            raise ValueError("Missing checkpoint for AnimateDiff")
        
        builder = ComfyWorkflowBuilder()
        
        # Load checkpoint
        builder.add_node("1", "CheckpointLoaderSimple",
            ckpt_name=checkpoint.filename)
        
        # Load AnimateDiff motion module
        builder.add_node("2", "ADE_AnimateDiffLoaderWithContext",
            model_name="mm_sd_v15_v2.safetensors",
            beta_schedule="sqrt_linear",
            context_options=None)
        
        # Apply motion module to model
        builder.add_node("3", "ADE_ApplyAnimateDiffModel",
            motion_model=("2", 0),
            model=("1", 0))
        
        # Text encoding
        builder.add_node("4", "CLIPTextEncode",
            text=request.prompt,
            clip=("1", 1))
        
        builder.add_node("5", "CLIPTextEncode",
            text=request.negative_prompt,
            clip=("1", 1))
        
        # Empty latent (batch = frames)
        builder.add_node("6", "EmptyLatentImage",
            width=request.width,
            height=request.height,
            batch_size=request.frames)
        
        # Sample
        builder.add_node("7", "KSampler",
            model=("3", 0),
            positive=("4", 0),
            negative=("5", 0),
            latent_image=("6", 0),
            seed=request.seed,
            steps=request.steps,
            cfg=7.5,
            sampler_name="euler_ancestral",
            scheduler="normal",
            denoise=1.0)
        
        # Decode
        builder.add_node("8", "VAEDecode",
            samples=("7", 0),
            vae=("1", 2))
        
        # Save as video
        builder.add_node("9", "SaveAnimatedWEBP",
            images=("8", 0),
            filename_prefix="vibe_animatediff",
            fps=float(request.fps),
            lossless=False,
            quality=85,
            method="default")
        
        return builder.build()


# =============================================================================
# FACTORY PATTERN — Strategy selection based on discovery
# =============================================================================

class StrategyFactory:
    """Factory for selecting the best video generation strategy.
    
    Uses discovery results to pick the optimal strategy based on:
    1. Available models
    2. Hardware constraints
    3. User preferences
    """
    
    # Registered strategies in priority order
    STRATEGIES: List[Type[VideoGenerationStrategy]] = [
        LTX2Strategy,      # Highest quality
        WanVideoStrategy,  # Good quality, I2V support
        AnimateDiffStrategy,  # Fallback, widely compatible
    ]
    
    @classmethod
    def create(
        cls,
        available_models: Dict[str, List[ModelSpec]],
        max_vram: float = 24.0,
        preference: str = "quality"  # "quality", "speed", "balanced"
    ) -> Optional[VideoGenerationStrategy]:
        """Create the best strategy for available models.
        
        Args:
            available_models: Dict of category -> list of ModelSpec
            max_vram: Hardware VRAM capacity (not free VRAM). 
                      Use total VRAM since models can be unloaded.
                      Default 24GB (typical high-end consumer GPU).
            preference: Optimization preference
            
        Returns:
            Best available strategy, or None if nothing works
        """
        candidates = []
        
        for strategy_class in cls.STRATEGIES:
            strategy = strategy_class()
            
            if strategy.can_use_models(available_models):
                models = strategy.select_models(available_models, max_vram)
                if models:
                    quality = strategy.quality_score(models)
                    candidates.append((strategy, models, quality))
        
        if not candidates:
            return None
        
        # Sort by preference
        if preference == "quality":
            candidates.sort(key=lambda x: x[2], reverse=True)
        elif preference == "speed":
            # Lower VRAM = faster
            candidates.sort(key=lambda x: sum(m.vram_required for m in x[1].values()))
        else:  # balanced
            candidates.sort(key=lambda x: x[2] / max(1, sum(m.vram_required for m in x[1].values())), reverse=True)
        
        return candidates[0][0]
    
    @classmethod
    def get_all_viable(
        cls,
        available_models: Dict[str, List[ModelSpec]],
        max_vram: float = 24.0
    ) -> List[VideoGenerationStrategy]:
        """Get all strategies that could work with available models."""
        viable = []
        
        for strategy_class in cls.STRATEGIES:
            strategy = strategy_class()
            if strategy.can_use_models(available_models):
                models = strategy.select_models(available_models, max_vram)
                if models:
                    viable.append(strategy)
        
        return viable


# =============================================================================
# OBSERVER PATTERN — Execution monitoring
# =============================================================================

class ExecutionObserver(Protocol):
    """Observer interface for workflow execution events."""
    
    def on_queued(self, prompt_id: str) -> None:
        """Called when workflow is queued."""
        ...
    
    def on_progress(self, prompt_id: str, node_id: str, progress: float) -> None:
        """Called during execution with progress updates."""
        ...
    
    def on_complete(self, prompt_id: str, result: WorkflowResult) -> None:
        """Called when execution completes."""
        ...
    
    def on_error(self, prompt_id: str, error: str) -> None:
        """Called when execution fails."""
        ...


class LoggingObserver:
    """Simple observer that logs events."""
    
    def on_queued(self, prompt_id: str) -> None:
        print(f"[QUEUED] {prompt_id}")
    
    def on_progress(self, prompt_id: str, node_id: str, progress: float) -> None:
        print(f"[PROGRESS] {prompt_id} - Node {node_id}: {progress:.1%}")
    
    def on_complete(self, prompt_id: str, result: WorkflowResult) -> None:
        status = "SUCCESS" if result.success else "FAILED"
        print(f"[{status}] {prompt_id} - {len(result.outputs)} outputs")
    
    def on_error(self, prompt_id: str, error: str) -> None:
        print(f"[ERROR] {prompt_id} - {error}")
