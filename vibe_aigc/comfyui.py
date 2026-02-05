"""ComfyUI backend for actual image/video generation.

This implements the paper's vision of AIGC - AI Generated Content,
not just text orchestration but actual multimodal content generation.
"""

import asyncio
import json
import uuid
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ComfyUIConfig:
    """Configuration for ComfyUI connection."""
    host: str = "127.0.0.1"
    port: int = 8188
    output_dir: Optional[str] = None  # Where to save generated images
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass 
class GenerationResult:
    """Result from a ComfyUI generation."""
    success: bool
    images: List[str] = field(default_factory=list)  # Paths or URLs to generated images
    prompt_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComfyUIBackend:
    """Backend for interacting with ComfyUI for image/video generation.
    
    This enables vibe-aigc to generate actual content, not just orchestrate.
    """
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self._client_id = str(uuid.uuid4())
        
    async def is_available(self) -> bool:
        """Check if ComfyUI is running and accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get ComfyUI system stats including GPU info."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.base_url}/system_stats") as resp:
                return await resp.json()
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models (checkpoints, loras, etc.)."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.base_url}/object_info") as resp:
                obj_info = await resp.json()
        
        models = {}
        
        # Extract checkpoint models
        if "CheckpointLoaderSimple" in obj_info:
            ckpt_info = obj_info["CheckpointLoaderSimple"]
            if "input" in ckpt_info and "required" in ckpt_info["input"]:
                ckpt_input = ckpt_info["input"]["required"].get("ckpt_name", [])
                if isinstance(ckpt_input, list) and len(ckpt_input) > 0:
                    if isinstance(ckpt_input[0], list):
                        models["checkpoints"] = ckpt_input[0]
        
        return models
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an image using ComfyUI's txt2img workflow.
        
        Args:
            prompt: Positive prompt describing what to generate
            negative_prompt: What to avoid in the generation
            width: Image width
            height: Image height  
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            seed: Random seed (None for random)
            checkpoint: Model checkpoint to use (None for default)
            
        Returns:
            GenerationResult with paths to generated images
        """
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        # Build the workflow
        workflow = self._build_txt2img_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            checkpoint=checkpoint
        )
        
        return await self._execute_workflow(workflow)
    
    async def generate_image_from_image(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float = 0.75,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Generate an image using img2img workflow.
        
        Args:
            image_path: Path to input image
            prompt: Positive prompt
            negative_prompt: Negative prompt
            denoise: Denoising strength (0-1, higher = more change)
            steps: Sampling steps
            cfg: Guidance scale
            seed: Random seed
            
        Returns:
            GenerationResult with paths to generated images
        """
        # TODO: Implement img2img workflow
        # This requires uploading the image first via /upload/image
        raise NotImplementedError("img2img not yet implemented")
    
    def _build_txt2img_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a txt2img workflow in ComfyUI API format."""
        
        # Default checkpoint if not specified
        if checkpoint is None:
            checkpoint = "v1-5-pruned-emaonly.safetensors"
        
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": steps
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "vibe_aigc",
                    "images": ["8", 0]
                }
            }
        }
        
        return workflow
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> GenerationResult:
        """Execute a workflow and wait for completion."""
        prompt_id = str(uuid.uuid4())
        
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
                    prompt_id = result.get("prompt_id", prompt_id)
                
                # Wait for completion by polling history
                images = await self._wait_for_completion(session, prompt_id)
                
                return GenerationResult(
                    success=True,
                    images=images,
                    prompt_id=prompt_id,
                    metadata={"workflow": workflow}
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
        timeout: float = 300,
        poll_interval: float = 0.5
    ) -> List[str]:
        """Wait for a prompt to complete and return output images."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with session.get(f"{self.config.base_url}/history/{prompt_id}") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    
                    if prompt_id in history:
                        prompt_data = history[prompt_id]
                        
                        # Check if completed
                        if prompt_data.get("status", {}).get("completed", False):
                            # Extract image paths from outputs
                            images = []
                            outputs = prompt_data.get("outputs", {})
                            
                            for node_id, node_output in outputs.items():
                                if "images" in node_output:
                                    for img in node_output["images"]:
                                        filename = img.get("filename")
                                        subfolder = img.get("subfolder", "")
                                        if filename:
                                            # Build the view URL
                                            img_url = f"{self.config.base_url}/view?filename={filename}"
                                            if subfolder:
                                                img_url += f"&subfolder={subfolder}"
                                            images.append(img_url)
                            
                            return images
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Generation timed out after {timeout}s")
    
    async def download_image(self, url: str, output_path: str) -> str:
        """Download a generated image to a local file."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(content)
                    return output_path
                else:
                    raise RuntimeError(f"Failed to download image: {resp.status}")


# Integration with vibe-aigc tools system
from .tools import BaseTool, ToolResult, ToolSpec, ToolCategory


class ComfyUIImageTool(BaseTool):
    """Tool for generating images via ComfyUI.
    
    This is the actual content generation that makes vibe-aigc
    a true AIGC system, not just text orchestration.
    """
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.backend = ComfyUIBackend(config)
        self._spec = ToolSpec(
            name="comfyui_image",
            description="Generate images using ComfyUI (local Stable Diffusion)",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Image generation prompt"},
                    "negative_prompt": {"type": "string", "description": "What to avoid"},
                    "width": {"type": "integer", "default": 512},
                    "height": {"type": "integer", "default": 512},
                    "steps": {"type": "integer", "default": 20},
                    "cfg": {"type": "number", "default": 7.0},
                    "seed": {"type": "integer", "description": "Random seed"},
                    "checkpoint": {"type": "string", "description": "Model checkpoint"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "images": {"type": "array", "items": {"type": "string"}},
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
        """Execute image generation.
        
        Args:
            inputs: Must contain 'prompt', optionally:
                - negative_prompt
                - width, height
                - steps, cfg
                - seed
                - checkpoint
        """
        prompt = inputs.get("prompt", "")
        if not prompt:
            return ToolResult(
                success=False,
                output=None,
                error="No prompt provided"
            )
        
        # Check if ComfyUI is available
        if not await self.backend.is_available():
            return ToolResult(
                success=False,
                output=None,
                error="ComfyUI is not running. Start it at http://127.0.0.1:8188"
            )
        
        result = await self.backend.generate_image(
            prompt=prompt,
            negative_prompt=inputs.get("negative_prompt", ""),
            width=inputs.get("width", 512),
            height=inputs.get("height", 512),
            steps=inputs.get("steps", 20),
            cfg=inputs.get("cfg", 7.0),
            seed=inputs.get("seed"),
            checkpoint=inputs.get("checkpoint")
        )
        
        if result.success:
            return ToolResult(
                success=True,
                output={
                    "images": result.images,
                    "prompt_id": result.prompt_id
                },
                metadata=result.metadata
            )
        else:
            return ToolResult(
                success=False,
                output=None,
                error=result.error
            )


def create_comfyui_registry(config: Optional[ComfyUIConfig] = None):
    """Create a tool registry with ComfyUI tools.
    
    This adds actual content generation capabilities to vibe-aigc.
    """
    from .tools import ToolRegistry
    
    registry = ToolRegistry()
    registry.register(ComfyUIImageTool(config))
    
    return registry
