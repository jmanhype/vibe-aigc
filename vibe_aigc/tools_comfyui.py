"""
ComfyUI Generation Tools - Image and Video Generation via VibeBackend.

Integrates the VibeBackend into the Paper's tool architecture (Section 5.4):
"The Planner traverses the system's atomic tool library...to select
the optimal ensemble of components."

This makes ComfyUI-based generation available to MetaPlanner as atomic tools.

Image Manipulation Tools:
- UpscaleTool: Upscale images using RealESRGAN, etc.
- InpaintTool: Mask-based inpainting
- Img2ImgTool: Image variation/transformation
- RemoveBackgroundTool: Background removal with alpha
- FaceRestoreTool: Face enhancement (CodeFormer, GFPGAN)
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .tools import BaseTool, ToolSpec, ToolResult, ToolCategory
from .vibe_backend import VibeBackend, GenerationRequest, GenerationResult
from .discovery import Capability


class ComfyUIExecutor:
    """Lightweight executor for ComfyUI workflows."""
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.url = comfyui_url.rstrip('/')
    
    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload an image to ComfyUI and return the filename."""
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            content_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'
            form.add_field('image', image_data, filename=filename, content_type=content_type)
            
            async with session.post(f"{self.url}/upload/image", data=form) as resp:
                result = await resp.json()
                return result.get("name", filename)
    
    async def download_image(self, url: str) -> bytes:
        """Download an image from a URL."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    return await resp.read()
                raise Exception(f"Failed to download image: {resp.status}")
    
    async def execute(self, workflow: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        """Execute a workflow and return the result."""
        try:
            async with aiohttp.ClientSession() as session:
                # Queue the workflow
                async with session.post(
                    f"{self.url}/prompt",
                    json={"prompt": workflow},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    
                    if "error" in data:
                        return {"success": False, "error": str(data["error"])}
                    
                    prompt_id = data["prompt_id"]
                
                # Poll for completion
                for _ in range(timeout // 2):
                    await asyncio.sleep(2)
                    
                    async with session.get(f"{self.url}/history/{prompt_id}") as resp:
                        history = await resp.json()
                        
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            
                            if status.get("completed") or status.get("status_str") == "success":
                                outputs = history[prompt_id].get("outputs", {})
                                for node_output in outputs.values():
                                    if "images" in node_output:
                                        img = node_output["images"][0]
                                        filename = img.get("filename", "")
                                        subfolder = img.get("subfolder", "")
                                        
                                        url = f"{self.url}/view?filename={filename}"
                                        if subfolder:
                                            url += f"&subfolder={subfolder}"
                                        
                                        return {
                                            "success": True,
                                            "output_url": url,
                                            "filename": filename
                                        }
                                
                                return {"success": False, "error": "No image output found"}
                            
                            if status.get("status_str") == "error":
                                return {"success": False, "error": "Workflow execution failed"}
                
                return {"success": False, "error": "Timeout waiting for result"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


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


class UpscaleTool(BaseTool):
    """
    Image upscaling tool using ComfyUI.
    
    Uses upscale models like RealESRGAN, 4x-UltraSharp, etc.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._executor = ComfyUIExecutor(comfyui_url)
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="upscale",
            description="Upscale images using AI models (RealESRGAN, 4x-UltraSharp, etc.)",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the image to upscale"},
                    "scale": {"type": "integer", "enum": [2, 4], "default": 2, "description": "Upscale factor (2x or 4x)"},
                    "model": {"type": "string", "default": "RealESRGAN_x4plus.pth", "description": "Upscale model to use"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "upscaled_url": {"type": "string", "description": "URL of the upscaled image"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://...", "scale": 4},
                    "output": {"upscaled_url": "http://..."}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Upscale an image."""
        try:
            image_url = inputs.get("image_url")
            scale = inputs.get("scale", 2)
            model = inputs.get("model", "RealESRGAN_x4plus.pth" if scale == 4 else "RealESRGAN_x2plus.pth")
            
            # Download and upload the image
            image_data = await self._executor.download_image(image_url)
            uploaded_name = await self._executor.upload_image(image_data, "upscale_input.png")
            
            # Build upscale workflow
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_name}
                },
                "2": {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": model}
                },
                "3": {
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": {
                        "upscale_model": ["2", 0],
                        "image": ["1", 0]
                    }
                },
                "4": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["3", 0],
                        "filename_prefix": "upscaled"
                    }
                }
            }
            
            result = await self._executor.execute(workflow)
            
            if result.get("success"):
                return ToolResult(
                    success=True,
                    output={"upscaled_url": result["output_url"]},
                    metadata={"model": model, "scale": scale}
                )
            else:
                return ToolResult(success=False, output=None, error=result.get("error"))
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class InpaintTool(BaseTool):
    """
    Mask-based inpainting tool using ComfyUI.
    
    Uses SD inpainting to fill masked regions based on a prompt.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._executor = ComfyUIExecutor(comfyui_url)
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="inpaint",
            description="Inpaint masked regions of an image based on a prompt",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["image_url", "mask_url", "prompt"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the image to inpaint"},
                    "mask_url": {"type": "string", "description": "URL of the mask (white = inpaint area)"},
                    "prompt": {"type": "string", "description": "What to generate in the masked area"},
                    "negative_prompt": {"type": "string", "description": "What to avoid"},
                    "strength": {"type": "number", "default": 1.0, "description": "Denoising strength (0-1)"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "inpainted_url": {"type": "string", "description": "URL of the inpainted image"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://...", "mask_url": "http://...", "prompt": "a red rose"},
                    "output": {"inpainted_url": "http://..."}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Inpaint an image region."""
        try:
            image_url = inputs.get("image_url")
            mask_url = inputs.get("mask_url")
            prompt = inputs.get("prompt", "")
            negative_prompt = inputs.get("negative_prompt", "blurry, distorted, ugly")
            strength = inputs.get("strength", 1.0)
            
            # Download and upload images
            image_data = await self._executor.download_image(image_url)
            mask_data = await self._executor.download_image(mask_url)
            
            uploaded_image = await self._executor.upload_image(image_data, "inpaint_image.png")
            uploaded_mask = await self._executor.upload_image(mask_data, "inpaint_mask.png")
            
            # Build inpaint workflow
            workflow = {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
                },
                "2": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_image}
                },
                "3": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_mask}
                },
                "4": {
                    "class_type": "ImageToMask",
                    "inputs": {"image": ["3", 0], "channel": "red"}
                },
                "5": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}
                },
                "6": {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {"samples": ["5", 0], "mask": ["4", 0]}
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": prompt, "clip": ["1", 1]}
                },
                "8": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": negative_prompt, "clip": ["1", 1]}
                },
                "9": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": 0,
                        "steps": 25,
                        "cfg": 7.0,
                        "sampler_name": "euler_ancestral",
                        "scheduler": "normal",
                        "denoise": strength,
                        "model": ["1", 0],
                        "positive": ["7", 0],
                        "negative": ["8", 0],
                        "latent_image": ["6", 0]
                    }
                },
                "10": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["9", 0], "vae": ["1", 2]}
                },
                "11": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["10", 0], "filename_prefix": "inpainted"}
                }
            }
            
            result = await self._executor.execute(workflow)
            
            if result.get("success"):
                return ToolResult(
                    success=True,
                    output={"inpainted_url": result["output_url"]},
                    metadata={"prompt": prompt}
                )
            else:
                return ToolResult(success=False, output=None, error=result.get("error"))
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class Img2ImgTool(BaseTool):
    """
    Image-to-image transformation tool using ComfyUI.
    
    Takes an input image and transforms it based on a prompt.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._executor = ComfyUIExecutor(comfyui_url)
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="img2img",
            description="Transform an image based on a text prompt",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["image_url", "prompt"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the input image"},
                    "prompt": {"type": "string", "description": "Transformation prompt"},
                    "negative_prompt": {"type": "string", "description": "What to avoid"},
                    "strength": {"type": "number", "default": 0.75, "description": "Transformation strength (0-1, higher = more change)"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "transformed_url": {"type": "string", "description": "URL of the transformed image"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://...", "prompt": "anime style", "strength": 0.7},
                    "output": {"transformed_url": "http://..."}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Transform an image based on a prompt."""
        try:
            image_url = inputs.get("image_url")
            prompt = inputs.get("prompt", "")
            negative_prompt = inputs.get("negative_prompt", "blurry, distorted, ugly, bad quality")
            strength = inputs.get("strength", 0.75)
            
            # Download and upload the image
            image_data = await self._executor.download_image(image_url)
            uploaded_name = await self._executor.upload_image(image_data, "img2img_input.png")
            
            # Build img2img workflow
            workflow = {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
                },
                "2": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_name}
                },
                "3": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}
                },
                "4": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": prompt, "clip": ["1", 1]}
                },
                "5": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": negative_prompt, "clip": ["1", 1]}
                },
                "6": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": 0,
                        "steps": 25,
                        "cfg": 7.0,
                        "sampler_name": "euler_ancestral",
                        "scheduler": "normal",
                        "denoise": strength,
                        "model": ["1", 0],
                        "positive": ["4", 0],
                        "negative": ["5", 0],
                        "latent_image": ["3", 0]
                    }
                },
                "7": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["6", 0], "vae": ["1", 2]}
                },
                "8": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["7", 0], "filename_prefix": "img2img"}
                }
            }
            
            result = await self._executor.execute(workflow)
            
            if result.get("success"):
                return ToolResult(
                    success=True,
                    output={"transformed_url": result["output_url"]},
                    metadata={"prompt": prompt, "strength": strength}
                )
            else:
                return ToolResult(success=False, output=None, error=result.get("error"))
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class RemoveBackgroundTool(BaseTool):
    """
    Background removal tool using ComfyUI.
    
    Removes background from images, outputting with alpha channel.
    Uses RMBG or similar segmentation models.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._executor = ComfyUIExecutor(comfyui_url)
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remove_background",
            description="Remove background from an image, output with transparency",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the image"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL of image with alpha channel"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://..."},
                    "output": {"image_url": "http://..."}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Remove background from an image."""
        try:
            image_url = inputs.get("image_url")
            
            # Download and upload the image
            image_data = await self._executor.download_image(image_url)
            uploaded_name = await self._executor.upload_image(image_data, "rmbg_input.png")
            
            # Build background removal workflow using RMBG node
            # This uses the ComfyUI-BRIA_AI-RMBG or similar node
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_name}
                },
                "2": {
                    "class_type": "BRIA_RMBG_ModelLoader",
                    "inputs": {}
                },
                "3": {
                    "class_type": "BRIA_RMBG_Zho",
                    "inputs": {
                        "rmbg_model": ["2", 0],
                        "image": ["1", 0]
                    }
                },
                "4": {
                    "class_type": "JoinImageWithAlpha",
                    "inputs": {
                        "image": ["1", 0],
                        "alpha": ["3", 0]
                    }
                },
                "5": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["4", 0],
                        "filename_prefix": "rmbg"
                    }
                }
            }
            
            result = await self._executor.execute(workflow)
            
            if result.get("success"):
                return ToolResult(
                    success=True,
                    output={"image_url": result["output_url"]},
                    metadata={}
                )
            else:
                # Fallback: try alternative node structure (InspyrenetRembg)
                workflow_alt = {
                    "1": {
                        "class_type": "LoadImage",
                        "inputs": {"image": uploaded_name}
                    },
                    "2": {
                        "class_type": "InspyrenetRembg",
                        "inputs": {
                            "image": ["1", 0],
                            "torchscript_jit": "default"
                        }
                    },
                    "3": {
                        "class_type": "SaveImage",
                        "inputs": {
                            "images": ["2", 0],
                            "filename_prefix": "rmbg"
                        }
                    }
                }
                
                result = await self._executor.execute(workflow_alt)
                
                if result.get("success"):
                    return ToolResult(
                        success=True,
                        output={"image_url": result["output_url"]},
                        metadata={}
                    )
                else:
                    return ToolResult(success=False, output=None, error=result.get("error"))
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FaceRestoreTool(BaseTool):
    """
    Face restoration/enhancement tool using ComfyUI.
    
    Uses CodeFormer or GFPGAN to enhance faces in images.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self._executor = ComfyUIExecutor(comfyui_url)
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="face_restore",
            description="Enhance and restore faces in an image using CodeFormer or GFPGAN",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the image with faces"},
                    "model": {"type": "string", "enum": ["codeformer", "gfpgan"], "default": "codeformer"},
                    "fidelity": {"type": "number", "default": 0.5, "description": "CodeFormer fidelity (0-1, higher = more faithful to original)"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "restored_url": {"type": "string", "description": "URL of the face-restored image"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://...", "model": "codeformer"},
                    "output": {"restored_url": "http://..."}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Restore faces in an image."""
        try:
            image_url = inputs.get("image_url")
            model = inputs.get("model", "codeformer")
            fidelity = inputs.get("fidelity", 0.5)
            
            # Download and upload the image
            image_data = await self._executor.download_image(image_url)
            uploaded_name = await self._executor.upload_image(image_data, "face_input.png")
            
            if model == "codeformer":
                # CodeFormer workflow
                workflow = {
                    "1": {
                        "class_type": "LoadImage",
                        "inputs": {"image": uploaded_name}
                    },
                    "2": {
                        "class_type": "FaceRestoreModelLoader",
                        "inputs": {"model_name": "codeformer-v0.1.0.pth"}
                    },
                    "3": {
                        "class_type": "FaceRestoreWithModel",
                        "inputs": {
                            "facerestore_model": ["2", 0],
                            "image": ["1", 0],
                            "fidelity": fidelity
                        }
                    },
                    "4": {
                        "class_type": "SaveImage",
                        "inputs": {
                            "images": ["3", 0],
                            "filename_prefix": "face_restored"
                        }
                    }
                }
            else:
                # GFPGAN workflow
                workflow = {
                    "1": {
                        "class_type": "LoadImage",
                        "inputs": {"image": uploaded_name}
                    },
                    "2": {
                        "class_type": "FaceRestoreModelLoader",
                        "inputs": {"model_name": "GFPGANv1.4.pth"}
                    },
                    "3": {
                        "class_type": "FaceRestoreWithModel",
                        "inputs": {
                            "facerestore_model": ["2", 0],
                            "image": ["1", 0],
                            "fidelity": 1.0  # GFPGAN doesn't use fidelity the same way
                        }
                    },
                    "4": {
                        "class_type": "SaveImage",
                        "inputs": {
                            "images": ["3", 0],
                            "filename_prefix": "face_restored"
                        }
                    }
                }
            
            result = await self._executor.execute(workflow)
            
            if result.get("success"):
                return ToolResult(
                    success=True,
                    output={"restored_url": result["output_url"]},
                    metadata={"model": model, "fidelity": fidelity}
                )
            else:
                return ToolResult(success=False, output=None, error=result.get("error"))
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


def create_comfyui_tools(comfyui_url: str = "http://127.0.0.1:8188") -> list:
    """Create ComfyUI-based generation and manipulation tools."""
    return [
        # Generation tools
        ImageGenerationTool(comfyui_url=comfyui_url),
        VideoGenerationTool(comfyui_url=comfyui_url),
        # Image manipulation tools
        UpscaleTool(comfyui_url=comfyui_url),
        InpaintTool(comfyui_url=comfyui_url),
        Img2ImgTool(comfyui_url=comfyui_url),
        RemoveBackgroundTool(comfyui_url=comfyui_url),
        FaceRestoreTool(comfyui_url=comfyui_url),
    ]
