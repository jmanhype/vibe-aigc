"""Video generation backend using AnimateDiff.

This extends vibe-aigc from images to video, completing the 
multimodal content generation capabilities.
"""

import asyncio
import json
import uuid
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .comfyui import ComfyUIConfig, GenerationResult
from .tools import BaseTool, ToolResult, ToolSpec, ToolCategory


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    frames: int = 16  # Number of frames (16, 24, or 32 typical)
    fps: int = 8  # Frames per second
    motion_scale: float = 1.0  # Motion intensity (0.5-1.5)
    loop: bool = False  # Whether to make it loop


class AnimateDiffBackend:
    """Backend for video generation via AnimateDiff in ComfyUI."""
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self._client_id = str(uuid.uuid4())
    
    async def is_available(self) -> bool:
        """Check if ComfyUI is running with AnimateDiff nodes."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/object_info",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return False
                    obj_info = await resp.json()
                    # Check for AnimateDiff nodes
                    return "ADE_AnimateDiffLoaderGen1" in obj_info or "AnimateDiffLoaderV1" in obj_info
        except Exception:
            return False
    
    async def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        frames: int = 16,
        fps: int = 8,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        motion_model: str = "v3_sd15_mm.ckpt",
        checkpoint: str = "v1-5-pruned-emaonly.safetensors"
    ) -> GenerationResult:
        """Generate a video using AnimateDiff.
        
        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid
            width: Video width
            height: Video height
            frames: Number of frames (16, 24, 32)
            fps: Frames per second for output
            steps: Sampling steps
            cfg: Guidance scale
            seed: Random seed
            motion_model: AnimateDiff motion model
            checkpoint: SD checkpoint to use
            
        Returns:
            GenerationResult with video file paths
        """
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        workflow = self._build_animatediff_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
            motion_model=motion_model,
            checkpoint=checkpoint,
            fps=fps
        )
        
        return await self._execute_workflow(workflow)
    
    def _build_animatediff_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        frames: int,
        steps: int,
        cfg: float,
        seed: int,
        motion_model: str,
        checkpoint: str,
        fps: int
    ) -> Dict[str, Any]:
        """Build AnimateDiff workflow in ComfyUI API format.
        
        Uses ADE_UseEvolvedSampling for proper AnimateDiff integration.
        """
        
        workflow = {
            # Load checkpoint
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint
                }
            },
            # Load AnimateDiff motion model
            "2": {
                "class_type": "ADE_LoadAnimateDiffModel",
                "inputs": {
                    "model_name": motion_model
                }
            },
            # Convert motion model to M_MODELS type
            "2b": {
                "class_type": "ADE_ApplyAnimateDiffModelSimple",
                "inputs": {
                    "motion_model": ["2", 0]
                }
            },
            # Apply AnimateDiff with evolved sampling
            "3": {
                "class_type": "ADE_UseEvolvedSampling",
                "inputs": {
                    "model": ["1", 0],
                    "m_models": ["2b", 0],
                    "beta_schedule": "sqrt_linear (AnimateDiff)"
                }
            },
            # Positive prompt
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 1],
                    "text": prompt
                }
            },
            # Negative prompt
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 1],
                    "text": negative_prompt
                }
            },
            # Empty latent batch for video frames
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": frames,
                    "height": height,
                    "width": width
                }
            },
            # KSampler with AnimateDiff model
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1,
                    "latent_image": ["6", 0],
                    "model": ["3", 0],
                    "negative": ["5", 0],
                    "positive": ["4", 0],
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": steps
                }
            },
            # VAE Decode
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["1", 2]
                }
            },
            # Use AnimateDiff's built-in combiner for GIF output
            "9": {
                "class_type": "ADE_AnimateDiffCombine",
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": fps,
                    "loop_count": 0,
                    "format": "image/gif",
                    "pingpong": False,
                    "save_image": True,
                    "filename_prefix": "vibe_aigc_video"
                }
            }
        }
        
        return workflow
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> GenerationResult:
        """Execute workflow and wait for completion."""
        payload = {
            "prompt": workflow,
            "client_id": self._client_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Queue the prompt
                async with session.post(
                    f"{self.config.base_url}/prompt",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return GenerationResult(
                            success=False,
                            error=f"Failed to queue prompt: {error_text}"
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id", "")
                
                # Wait for completion
                videos = await self._wait_for_completion(session, prompt_id)
                
                return GenerationResult(
                    success=True,
                    images=videos,  # reusing images field for video paths
                    prompt_id=prompt_id,
                    metadata={"type": "video", "workflow": "animatediff"}
                )
                
        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e)
            )
    
    async def _wait_for_completion(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
        timeout: float = 600,  # Videos take longer
        poll_interval: float = 1.0
    ) -> List[str]:
        """Wait for video generation to complete."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with session.get(f"{self.config.base_url}/history/{prompt_id}") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    
                    if prompt_id in history:
                        prompt_data = history[prompt_id]
                        
                        if prompt_data.get("status", {}).get("completed", False):
                            videos = []
                            outputs = prompt_data.get("outputs", {})
                            
                            for node_id, node_output in outputs.items():
                                # Check for animated outputs (webp, gif)
                                if "images" in node_output:
                                    for img in node_output["images"]:
                                        filename = img.get("filename")
                                        subfolder = img.get("subfolder", "")
                                        if filename:
                                            # Animated webp/gif files
                                            url = f"{self.config.base_url}/view?filename={filename}"
                                            if subfolder:
                                                url += f"&subfolder={subfolder}"
                                            videos.append(url)
                                # Also check gifs field (some nodes use this)
                                if "gifs" in node_output:
                                    for gif in node_output["gifs"]:
                                        filename = gif.get("filename")
                                        subfolder = gif.get("subfolder", "")
                                        if filename:
                                            url = f"{self.config.base_url}/view?filename={filename}"
                                            if subfolder:
                                                url += f"&subfolder={subfolder}"
                                            videos.append(url)
                            
                            return videos
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Video generation timed out after {timeout}s")


class AnimateDiffTool(BaseTool):
    """Tool for generating videos via AnimateDiff."""
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.backend = AnimateDiffBackend(config)
        self._spec = ToolSpec(
            name="animatediff_video",
            description="Generate short video clips using AnimateDiff (local)",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Video description"},
                    "negative_prompt": {"type": "string"},
                    "width": {"type": "integer", "default": 512},
                    "height": {"type": "integer", "default": 512},
                    "frames": {"type": "integer", "default": 16, "description": "Number of frames (16, 24, 32)"},
                    "fps": {"type": "integer", "default": 8},
                    "steps": {"type": "integer", "default": 20},
                    "cfg": {"type": "number", "default": 7.0},
                    "seed": {"type": "integer"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "videos": {"type": "array", "items": {"type": "string"}},
                    "prompt_id": {"type": "string"}
                }
            }
        )
    
    @property
    def spec(self) -> ToolSpec:
        return self._spec
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute video generation."""
        prompt = inputs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, output=None, error="No prompt provided")
        
        # Check availability
        if not await self.backend.is_available():
            return ToolResult(
                success=False,
                output=None,
                error="AnimateDiff not available. Install ComfyUI-AnimateDiff-Evolved node."
            )
        
        result = await self.backend.generate_video(
            prompt=prompt,
            negative_prompt=inputs.get("negative_prompt", ""),
            width=inputs.get("width", 512),
            height=inputs.get("height", 512),
            frames=inputs.get("frames", 16),
            fps=inputs.get("fps", 8),
            steps=inputs.get("steps", 20),
            cfg=inputs.get("cfg", 7.0),
            seed=inputs.get("seed")
        )
        
        if result.success:
            return ToolResult(
                success=True,
                output={
                    "videos": result.images,
                    "prompt_id": result.prompt_id
                },
                metadata=result.metadata
            )
        else:
            return ToolResult(success=False, output=None, error=result.error)
