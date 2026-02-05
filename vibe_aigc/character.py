"""Character consistency using IPAdapter.

This enables maintaining character appearance across multiple
generations - critical for music videos, stories, etc.
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
class CharacterReference:
    """A character reference for consistency."""
    name: str
    reference_image: str  # Path or URL to reference image
    description: str = ""
    weight: float = 0.8  # How strongly to apply the reference (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "reference_image": self.reference_image,
            "description": self.description,
            "weight": self.weight
        }


class CharacterBank:
    """Bank of character references for consistency.
    
    Like the paper's Character Bank - maintains character identity
    across multiple generations.
    """
    
    def __init__(self):
        self._characters: Dict[str, CharacterReference] = {}
    
    def add(self, character: CharacterReference) -> None:
        """Add a character to the bank."""
        self._characters[character.name.lower()] = character
    
    def get(self, name: str) -> Optional[CharacterReference]:
        """Get a character by name."""
        return self._characters.get(name.lower())
    
    def list_characters(self) -> List[str]:
        """List all character names."""
        return list(self._characters.keys())
    
    def remove(self, name: str) -> bool:
        """Remove a character from the bank."""
        if name.lower() in self._characters:
            del self._characters[name.lower()]
            return True
        return False


class IPAdapterBackend:
    """Backend for character-consistent generation using IPAdapter."""
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self._client_id = str(uuid.uuid4())
    
    async def is_available(self) -> bool:
        """Check if IPAdapter nodes are available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/object_info",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return False
                    obj_info = await resp.json()
                    # Check for IPAdapter nodes
                    return "IPAdapterApply" in obj_info or "IPAdapter" in obj_info
        except Exception:
            return False
    
    async def generate_with_reference(
        self,
        prompt: str,
        reference_image: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        reference_weight: float = 0.8,
        checkpoint: str = "v1-5-pruned-emaonly.safetensors"
    ) -> GenerationResult:
        """Generate an image consistent with a reference.
        
        Args:
            prompt: Text description
            reference_image: Path to reference image for consistency
            negative_prompt: What to avoid
            width: Output width
            height: Output height
            steps: Sampling steps
            cfg: Guidance scale
            seed: Random seed
            reference_weight: How strongly to apply reference (0-1)
            checkpoint: SD checkpoint
            
        Returns:
            GenerationResult with consistent image
        """
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        # First, upload the reference image if it's a local path
        if not reference_image.startswith(('http://', 'https://')):
            reference_image = await self._upload_image(reference_image)
        
        workflow = self._build_ipadapter_workflow(
            prompt=prompt,
            reference_image=reference_image,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            reference_weight=reference_weight,
            checkpoint=checkpoint
        )
        
        return await self._execute_workflow(workflow)
    
    async def _upload_image(self, image_path: str) -> str:
        """Upload a local image to ComfyUI."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('image',
                          open(path, 'rb'),
                          filename=path.name,
                          content_type='image/png')
            
            async with session.post(
                f"{self.config.base_url}/upload/image",
                data=data
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to upload image: {await resp.text()}")
                result = await resp.json()
                return result.get("name", path.name)
    
    def _build_ipadapter_workflow(
        self,
        prompt: str,
        reference_image: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        reference_weight: float,
        checkpoint: str
    ) -> Dict[str, Any]:
        """Build IPAdapter workflow for character-consistent generation."""
        
        workflow = {
            # Load checkpoint
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint
                }
            },
            # Load reference image
            "2": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": reference_image
                }
            },
            # Load IPAdapter model
            "3": {
                "class_type": "IPAdapterModelLoader",
                "inputs": {
                    "ipadapter_file": "ip-adapter_sd15.safetensors"
                }
            },
            # Load CLIP Vision
            "4": {
                "class_type": "CLIPVisionLoader",
                "inputs": {
                    "clip_name": "clip_vision_g.safetensors"
                }
            },
            # Apply IPAdapter
            "5": {
                "class_type": "IPAdapterApply",
                "inputs": {
                    "model": ["1", 0],
                    "ipadapter": ["3", 0],
                    "clip_vision": ["4", 0],
                    "image": ["2", 0],
                    "weight": reference_weight,
                    "noise": 0.0,
                    "weight_type": "standard"
                }
            },
            # Positive prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 1],
                    "text": prompt
                }
            },
            # Negative prompt
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 1],
                    "text": negative_prompt
                }
            },
            # Empty latent
            "8": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width
                }
            },
            # KSampler with IPAdapter-enhanced model
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1,
                    "latent_image": ["8", 0],
                    "model": ["5", 0],  # Use IPAdapter-enhanced model
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": steps
                }
            },
            # VAE Decode
            "10": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["1", 2]
                }
            },
            # Save image
            "11": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "vibe_aigc_character",
                    "images": ["10", 0]
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
                
                images = await self._wait_for_completion(session, prompt_id)
                
                return GenerationResult(
                    success=True,
                    images=images,
                    prompt_id=prompt_id,
                    metadata={"type": "character_consistent"}
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
        """Wait for generation to complete."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with session.get(f"{self.config.base_url}/history/{prompt_id}") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    
                    if prompt_id in history:
                        prompt_data = history[prompt_id]
                        
                        if prompt_data.get("status", {}).get("completed", False):
                            images = []
                            outputs = prompt_data.get("outputs", {})
                            
                            for node_id, node_output in outputs.items():
                                if "images" in node_output:
                                    for img in node_output["images"]:
                                        filename = img.get("filename")
                                        subfolder = img.get("subfolder", "")
                                        if filename:
                                            url = f"{self.config.base_url}/view?filename={filename}"
                                            if subfolder:
                                                url += f"&subfolder={subfolder}"
                                            images.append(url)
                            
                            return images
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Generation timed out after {timeout}s")


class CharacterConsistentTool(BaseTool):
    """Tool for generating character-consistent images."""
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.backend = IPAdapterBackend(config)
        self.character_bank = CharacterBank()
        self._spec = ToolSpec(
            name="character_consistent",
            description="Generate images with character consistency using IPAdapter",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string"},
                    "character_name": {"type": "string", "description": "Name of character from bank"},
                    "reference_image": {"type": "string", "description": "Direct path to reference image"},
                    "reference_weight": {"type": "number", "default": 0.8},
                    "negative_prompt": {"type": "string"},
                    "width": {"type": "integer", "default": 512},
                    "height": {"type": "integer", "default": 512},
                    "steps": {"type": "integer", "default": 20},
                    "cfg": {"type": "number", "default": 7.0}
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
    
    def add_character(self, character: CharacterReference) -> None:
        """Add a character to the bank."""
        self.character_bank.add(character)
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute character-consistent generation."""
        prompt = inputs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, output=None, error="No prompt provided")
        
        # Get reference image
        reference_image = inputs.get("reference_image")
        character_name = inputs.get("character_name")
        reference_weight = inputs.get("reference_weight", 0.8)
        
        if character_name:
            char = self.character_bank.get(character_name)
            if char:
                reference_image = char.reference_image
                reference_weight = char.weight
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Character '{character_name}' not found in bank"
                )
        
        if not reference_image:
            return ToolResult(
                success=False,
                output=None,
                error="No reference_image or character_name provided"
            )
        
        result = await self.backend.generate_with_reference(
            prompt=prompt,
            reference_image=reference_image,
            negative_prompt=inputs.get("negative_prompt", ""),
            width=inputs.get("width", 512),
            height=inputs.get("height", 512),
            steps=inputs.get("steps", 20),
            cfg=inputs.get("cfg", 7.0),
            reference_weight=reference_weight
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
            return ToolResult(success=False, output=None, error=result.error)
