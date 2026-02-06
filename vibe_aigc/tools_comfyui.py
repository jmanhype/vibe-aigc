"""
ComfyUI Generation Tools - Image and Video Generation via VibeBackend.

Integrates the VibeBackend into the Paper's tool architecture (Section 5.4):
"The Planner traverses the system's atomic tool library...to select
the optimal ensemble of components."

This makes ComfyUI-based generation available to MetaPlanner as atomic tools.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .tools import BaseTool, ToolSpec, ToolResult, ToolCategory
from .vibe_backend import VibeBackend, GenerationRequest, GenerationResult
from .discovery import Capability


class ImageGenerationTool(BaseTool):
    """
    Image generation tool using local ComfyUI.
    
    Maps high-level image requests to VibeBackend execution.
    MetaPlanner can use this for visual content generation.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._backend: Optional[VibeBackend] = None
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generation",
            description="Generate images using local AI models (FLUX, SD, etc.)",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Image description"},
                    "negative_prompt": {"type": "string", "description": "What to avoid"},
                    "style": {"type": "string", "description": "Visual style hints"},
                    "width": {"type": "integer", "default": 768},
                    "height": {"type": "integer", "default": 512},
                    "steps": {"type": "integer", "default": 20},
                    "cfg": {"type": "number", "default": 7.0}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "image_url": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "feedback": {"type": "string"}
                }
            },
            examples=[
                {
                    "input": {"prompt": "cyberpunk cityscape at night, neon lights, rain"},
                    "output": {"image_url": "http://...", "quality_score": 7.5}
                }
            ]
        )
    
    async def _ensure_backend(self) -> VibeBackend:
        """Lazily initialize backend."""
        if self._backend is None:
            self._backend = VibeBackend(
                comfyui_url=self.comfyui_url,
                enable_vlm=True,
                max_attempts=2,
                quality_threshold=6.0
            )
            await self._backend.initialize()
        return self._backend
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate an image based on inputs."""
        try:
            backend = await self._ensure_backend()
            
            # Build prompt from inputs + context (knowledge base hints)
            prompt = inputs.get("prompt", "")
            style = inputs.get("style", "")
            
            # If knowledge base provided style hints, incorporate them
            if context and "technical_specs" in context:
                specs = context["technical_specs"]
                if "lighting" in specs:
                    prompt += f", {', '.join(specs['lighting'])}"
                if "color_grading" in specs:
                    prompt += f", {specs['color_grading']}"
            
            if style:
                prompt = f"{prompt}, {style}"
            
            result = await backend.generate(GenerationRequest(
                prompt=prompt,
                capability=Capability.TEXT_TO_IMAGE,
                negative_prompt=inputs.get("negative_prompt", ""),
                width=inputs.get("width", 768),
                height=inputs.get("height", 512),
                steps=inputs.get("steps", 20),
                cfg=inputs.get("cfg", 7.0)
            ))
            
            if result.success:
                return ToolResult(
                    success=True,
                    output={
                        "image_url": result.output_url,
                        "quality_score": result.quality_score,
                        "feedback": result.feedback,
                        "strengths": result.strengths,
                        "weaknesses": result.weaknesses
                    },
                    metadata={
                        "model_used": result.model_used,
                        "attempts": result.attempts
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=result.error
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class VideoGenerationTool(BaseTool):
    """
    Video generation tool using local ComfyUI.
    
    Uses the I2V pipeline: FLUX → Wan I2V animation.
    MetaPlanner can use this for motion content.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._backend: Optional[VibeBackend] = None
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_generation",
            description="Generate videos using local AI models (FLUX + Wan I2V)",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Video description"},
                    "negative_prompt": {"type": "string", "description": "What to avoid"},
                    "style": {"type": "string", "description": "Visual style hints"},
                    "motion": {"type": "string", "description": "Motion description"},
                    "width": {"type": "integer", "default": 832},
                    "height": {"type": "integer", "default": 480},
                    "frames": {"type": "integer", "default": 33},
                    "fps": {"type": "integer", "default": 16}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "video_url": {"type": "string"},
                    "duration_seconds": {"type": "number"}
                }
            },
            examples=[
                {
                    "input": {"prompt": "samurai walking through rain", "frames": 33},
                    "output": {"video_url": "http://...", "duration_seconds": 2.0}
                }
            ]
        )
    
    async def _ensure_backend(self) -> VibeBackend:
        """Lazily initialize backend."""
        if self._backend is None:
            self._backend = VibeBackend(
                comfyui_url=self.comfyui_url,
                enable_vlm=False,  # VLM slower for video
                max_attempts=1
            )
            await self._backend.initialize()
        return self._backend
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate a video based on inputs."""
        try:
            backend = await self._ensure_backend()
            
            # Build prompt
            prompt = inputs.get("prompt", "")
            style = inputs.get("style", "")
            motion = inputs.get("motion", "")
            
            # Incorporate knowledge base hints
            if context and "technical_specs" in context:
                specs = context["technical_specs"]
                if "camera" in specs:
                    prompt += f", {', '.join(specs['camera'][:2])}"
                if "editing" in specs:
                    motion = motion or specs["editing"][0] if specs["editing"] else ""
            
            if style:
                prompt = f"{prompt}, {style}"
            if motion:
                prompt = f"{prompt}, {motion}"
            
            frames = inputs.get("frames", 33)
            fps = inputs.get("fps", 16)
            
            result = await backend.generate(GenerationRequest(
                prompt=prompt,
                capability=Capability.TEXT_TO_VIDEO,
                negative_prompt=inputs.get("negative_prompt", ""),
                width=inputs.get("width", 832),
                height=inputs.get("height", 480),
                frames=frames,
                steps=20,
                cfg=5.0
            ))
            
            if result.success:
                return ToolResult(
                    success=True,
                    output={
                        "video_url": result.output_url,
                        "duration_seconds": frames / fps
                    },
                    metadata={
                        "frames": frames,
                        "fps": fps
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=result.error
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


def create_comfyui_tools(comfyui_url: str = "http://127.0.0.1:8188") -> list:
    """Create ComfyUI-based generation tools."""
    return [
        ImageGenerationTool(comfyui_url=comfyui_url),
        VideoGenerationTool(comfyui_url=comfyui_url)
    ]
