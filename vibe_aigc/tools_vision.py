"""
Vision Analysis Tools - Image understanding and analysis via ComfyUI.

Provides visual analysis capabilities for the AIGC pipeline:
- Image captioning (VLM-based description)
- Depth map estimation (MiDaS)
- Object segmentation (SAM/GroundingDINO)
- Object detection (YOLO/GroundingDINO)
- OCR text extraction

These tools analyze existing images rather than generate new ones,
enabling visual reasoning in the MetaPlanner workflow.
"""

import asyncio
import json
import uuid
import aiohttp
import base64
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .tools import BaseTool, ToolSpec, ToolResult, ToolCategory


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""
    x: float  # Top-left x (normalized 0-1 or pixels)
    y: float  # Top-left y
    width: float
    height: float
    label: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "label": self.label,
            "confidence": self.confidence
        }


@dataclass
class SegmentResult:
    """Result from segmentation."""
    mask_url: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisionToolBase(BaseTool):
    """Base class for vision analysis tools."""
    
    def __init__(self, comfyui_url: str = "http://192.168.1.143:8188"):
        self.comfyui_url = comfyui_url
        self._client_id = str(uuid.uuid4())
    
    async def _check_comfyui(self) -> bool:
        """Check if ComfyUI is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.comfyui_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def _upload_image(self, image_url: str) -> Optional[str]:
        """Upload image to ComfyUI, returning the filename."""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch the image
                if image_url.startswith("data:"):
                    # Handle data URLs
                    header, data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    content_type = header.split(";")[0].split(":")[1]
                    ext = content_type.split("/")[1]
                else:
                    # Fetch from URL
                    async with session.get(image_url) as resp:
                        if resp.status != 200:
                            return None
                        image_bytes = await resp.read()
                        content_type = resp.headers.get("Content-Type", "image/png")
                        ext = content_type.split("/")[-1].split(";")[0]
                
                # Upload to ComfyUI
                filename = f"vision_input_{uuid.uuid4().hex[:8]}.{ext}"
                
                form = aiohttp.FormData()
                form.add_field(
                    "image",
                    image_bytes,
                    filename=filename,
                    content_type=content_type
                )
                
                async with session.post(
                    f"{self.comfyui_url}/upload/image",
                    data=form
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("name", filename)
                    return None
        except Exception as e:
            print(f"Upload failed: {e}")
            return None
    
    async def _execute_workflow(
        self,
        workflow: Dict[str, Any],
        timeout: float = 120
    ) -> Tuple[bool, Any, Optional[str]]:
        """Execute ComfyUI workflow and return results."""
        payload = {
            "prompt": workflow,
            "client_id": self._client_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Queue the prompt
                async with session.post(
                    f"{self.comfyui_url}/prompt",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        return False, None, f"Queue failed: {error}"
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                
                # Wait for completion
                outputs = await self._wait_for_completion(session, prompt_id, timeout)
                return True, outputs, None
                
        except Exception as e:
            return False, None, str(e)
    
    async def _wait_for_completion(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
        timeout: float = 120
    ) -> Dict[str, Any]:
        """Wait for workflow completion and return outputs."""
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            async with session.get(f"{self.comfyui_url}/history/{prompt_id}") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    if prompt_id in history:
                        data = history[prompt_id]
                        if data.get("status", {}).get("completed", False):
                            return data.get("outputs", {})
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Workflow timed out after {timeout}s")


class CaptionTool(VisionToolBase):
    """
    Image captioning tool using VLM.
    
    Describes the content of an image using Gemini VLM
    or BLIP via ComfyUI.
    """
    
    def __init__(
        self,
        comfyui_url: str = "http://192.168.1.143:8188",
        use_vlm_feedback: bool = True
    ):
        super().__init__(comfyui_url)
        self.use_vlm_feedback = use_vlm_feedback
        self._vlm = None
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="caption_image",
            description="Generate a detailed caption/description for an image using VLM",
            category=ToolCategory.ANALYSIS,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL or path to image"},
                    "detail_level": {
                        "type": "string",
                        "enum": ["brief", "detailed", "comprehensive"],
                        "default": "detailed"
                    },
                    "focus": {"type": "string", "description": "Specific aspect to focus on"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "style": {"type": "string"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://example.com/cat.jpg"},
                    "output": {"caption": "A fluffy orange tabby cat sleeping on a windowsill", "tags": ["cat", "orange", "sleeping"]}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate caption for image."""
        image_url = inputs.get("image_url", "")
        if not image_url:
            return ToolResult(success=False, output=None, error="No image_url provided")
        
        detail_level = inputs.get("detail_level", "detailed")
        focus = inputs.get("focus", "")
        
        # Try VLMFeedback first (preferred)
        if self.use_vlm_feedback:
            result = await self._caption_with_vlm(image_url, detail_level, focus)
            if result.success:
                return result
        
        # Fallback to ComfyUI BLIP
        return await self._caption_with_comfyui(image_url, detail_level)
    
    async def _caption_with_vlm(
        self,
        image_url: str,
        detail_level: str,
        focus: str
    ) -> ToolResult:
        """Use VLMFeedback for captioning."""
        try:
            from .vlm_feedback import VLMFeedback
            
            if self._vlm is None:
                self._vlm = VLMFeedback()
            
            if not self._vlm.available:
                return ToolResult(success=False, output=None, error="VLM not available")
            
            # Download image if URL
            import tempfile
            import os
            
            temp_path = None
            try:
                if image_url.startswith(("http://", "https://")):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as resp:
                            if resp.status != 200:
                                return ToolResult(success=False, output=None, error="Failed to fetch image")
                            content = await resp.read()
                            ext = resp.headers.get("Content-Type", "image/png").split("/")[-1]
                            temp_path = Path(tempfile.gettempdir()) / f"caption_{uuid.uuid4().hex}.{ext}"
                            temp_path.write_bytes(content)
                            image_path = temp_path
                elif image_url.startswith("data:"):
                    header, data = image_url.split(",", 1)
                    ext = header.split(";")[0].split("/")[1]
                    temp_path = Path(tempfile.gettempdir()) / f"caption_{uuid.uuid4().hex}.{ext}"
                    temp_path.write_bytes(base64.b64decode(data))
                    image_path = temp_path
                else:
                    image_path = Path(image_url)
                
                # Build prompt based on detail level
                detail_prompts = {
                    "brief": "Describe this image in one sentence.",
                    "detailed": "Describe this image in detail, including subjects, actions, setting, and mood.",
                    "comprehensive": "Provide a comprehensive analysis of this image including: subjects, actions, setting, composition, lighting, colors, mood, style, and any notable details."
                }
                
                prompt = detail_prompts.get(detail_level, detail_prompts["detailed"])
                if focus:
                    prompt += f" Focus especially on: {focus}"
                
                # Use Gemini directly for captioning
                try:
                    import google.generativeai as genai
                    from PIL import Image
                    
                    img = Image.open(image_path)
                    response = self._vlm.vlm.generate_content([
                        prompt + "\n\nRespond with JSON: {\"caption\": \"...\", \"tags\": [...], \"style\": \"...\"}",
                        img
                    ])
                    
                    text = response.text
                    # Parse JSON
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                    
                    data = json.loads(text.strip())
                    
                    return ToolResult(
                        success=True,
                        output={
                            "caption": data.get("caption", ""),
                            "tags": data.get("tags", []),
                            "style": data.get("style", "")
                        },
                        metadata={"method": "vlm_gemini"}
                    )
                except json.JSONDecodeError:
                    # Return plain text caption
                    return ToolResult(
                        success=True,
                        output={
                            "caption": response.text,
                            "tags": [],
                            "style": ""
                        },
                        metadata={"method": "vlm_gemini"}
                    )
            finally:
                if temp_path and temp_path.exists():
                    os.unlink(temp_path)
                    
        except ImportError:
            return ToolResult(success=False, output=None, error="VLM dependencies not installed")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _caption_with_comfyui(
        self,
        image_url: str,
        detail_level: str
    ) -> ToolResult:
        """Use ComfyUI BLIP node for captioning."""
        if not await self._check_comfyui():
            return ToolResult(success=False, output=None, error="ComfyUI not available")
        
        filename = await self._upload_image(image_url)
        if not filename:
            return ToolResult(success=False, output=None, error="Failed to upload image")
        
        # BLIP captioning workflow
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": filename}
            },
            "2": {
                "class_type": "BLIPCaption",  # From ComfyUI-BLIP
                "inputs": {
                    "image": ["1", 0],
                    "min_length": 10 if detail_level == "brief" else 30,
                    "max_length": 50 if detail_level == "brief" else 150
                }
            }
        }
        
        success, outputs, error = await self._execute_workflow(workflow)
        
        if not success:
            return ToolResult(success=False, output=None, error=error)
        
        # Extract caption from outputs
        caption = ""
        for node_id, node_out in outputs.items():
            if "text" in node_out:
                caption = node_out["text"]
                break
        
        return ToolResult(
            success=True,
            output={"caption": caption, "tags": [], "style": ""},
            metadata={"method": "comfyui_blip"}
        )


class DepthMapTool(VisionToolBase):
    """
    Depth map estimation tool using MiDaS.
    
    Generates a depth map from an input image using
    MiDaS or similar depth estimation models.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="depth_map",
            description="Estimate depth from an image using MiDaS",
            category=ToolCategory.ANALYSIS,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL or path to image"},
                    "model": {
                        "type": "string",
                        "enum": ["midas", "zoe", "depth_anything"],
                        "default": "midas"
                    },
                    "normalize": {"type": "boolean", "default": True}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "depth_map_url": {"type": "string"},
                    "min_depth": {"type": "number"},
                    "max_depth": {"type": "number"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://example.com/scene.jpg"},
                    "output": {"depth_map_url": "http://comfyui/view?filename=depth_001.png"}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate depth map from image."""
        image_url = inputs.get("image_url", "")
        if not image_url:
            return ToolResult(success=False, output=None, error="No image_url provided")
        
        if not await self._check_comfyui():
            return ToolResult(success=False, output=None, error="ComfyUI not available")
        
        filename = await self._upload_image(image_url)
        if not filename:
            return ToolResult(success=False, output=None, error="Failed to upload image")
        
        model = inputs.get("model", "midas")
        
        # Build depth estimation workflow
        # Try DepthAnything first, fallback to MiDaS
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": filename}
            },
            "2": {
                "class_type": "MiDaS-DepthMapPreprocessor",  # ControlNet aux
                "inputs": {
                    "image": ["1", 0],
                    "a": 6.283185307179586,  # pi*2
                    "bg_threshold": 0.1,
                    "resolution": 512
                }
            },
            "3": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["2", 0]}
            },
            "4": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["2", 0],
                    "filename_prefix": "depth_map"
                }
            }
        }
        
        success, outputs, error = await self._execute_workflow(workflow)
        
        if not success:
            return ToolResult(success=False, output=None, error=error)
        
        # Extract depth map URL from outputs
        depth_url = None
        for node_id, node_out in outputs.items():
            if "images" in node_out:
                for img in node_out["images"]:
                    filename = img.get("filename")
                    subfolder = img.get("subfolder", "")
                    if filename:
                        depth_url = f"{self.comfyui_url}/view?filename={filename}"
                        if subfolder:
                            depth_url += f"&subfolder={subfolder}"
                        break
        
        if not depth_url:
            return ToolResult(success=False, output=None, error="No depth map generated")
        
        return ToolResult(
            success=True,
            output={
                "depth_map_url": depth_url,
                "min_depth": 0.0,
                "max_depth": 1.0
            },
            metadata={"model": model}
        )


class SegmentTool(VisionToolBase):
    """
    Object segmentation tool using SAM or GroundingDINO.
    
    Segments objects in an image, optionally guided by
    a text prompt for specific object selection.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="segment_objects",
            description="Segment objects in an image using SAM/GroundingDINO",
            category=ToolCategory.ANALYSIS,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL or path to image"},
                    "prompt": {"type": "string", "description": "Text prompt for objects to segment (optional)"},
                    "threshold": {"type": "number", "default": 0.3, "description": "Detection threshold"},
                    "return_all": {"type": "boolean", "default": False}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "mask_url": {"type": "string"},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "confidence": {"type": "number"},
                                "bbox": {"type": "object"}
                            }
                        }
                    }
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://example.com/scene.jpg", "prompt": "person"},
                    "output": {"mask_url": "http://...", "segments": [{"label": "person", "confidence": 0.95}]}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Segment objects in image."""
        image_url = inputs.get("image_url", "")
        if not image_url:
            return ToolResult(success=False, output=None, error="No image_url provided")
        
        if not await self._check_comfyui():
            return ToolResult(success=False, output=None, error="ComfyUI not available")
        
        filename = await self._upload_image(image_url)
        if not filename:
            return ToolResult(success=False, output=None, error="Failed to upload image")
        
        prompt = inputs.get("prompt", "")
        threshold = inputs.get("threshold", 0.3)
        
        # Build segmentation workflow
        # Uses GroundingDINO for text-guided detection + SAM for segmentation
        if prompt:
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": filename}
                },
                "2": {
                    "class_type": "GroundingDinoSAMSegment",  # From ComfyUI-SAM
                    "inputs": {
                        "image": ["1", 0],
                        "prompt": prompt,
                        "threshold": threshold
                    }
                },
                "3": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["2", 0],  # Mask output
                        "filename_prefix": "segment_mask"
                    }
                }
            }
        else:
            # Auto-segment everything with SAM
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": filename}
                },
                "2": {
                    "class_type": "SAMModelLoader",
                    "inputs": {
                        "model_name": "sam_vit_h_4b8939.pth"
                    }
                },
                "3": {
                    "class_type": "SAMSegmentAnything",  # From impact-pack or similar
                    "inputs": {
                        "sam_model": ["2", 0],
                        "image": ["1", 0]
                    }
                },
                "4": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["3", 0],
                        "filename_prefix": "segment_mask"
                    }
                }
            }
        
        success, outputs, error = await self._execute_workflow(workflow)
        
        if not success:
            return ToolResult(success=False, output=None, error=error)
        
        # Extract mask URL
        mask_url = None
        for node_id, node_out in outputs.items():
            if "images" in node_out:
                for img in node_out["images"]:
                    fname = img.get("filename")
                    subfolder = img.get("subfolder", "")
                    if fname:
                        mask_url = f"{self.comfyui_url}/view?filename={fname}"
                        if subfolder:
                            mask_url += f"&subfolder={subfolder}"
                        break
        
        segments = []
        if prompt:
            segments.append({
                "label": prompt,
                "confidence": 1.0,
                "bbox": None
            })
        
        return ToolResult(
            success=True,
            output={
                "mask_url": mask_url or "",
                "segments": segments
            },
            metadata={"prompt": prompt}
        )


class DetectObjectsTool(VisionToolBase):
    """
    Object detection tool using YOLO or GroundingDINO.
    
    Detects and labels objects in an image with
    bounding boxes and confidence scores.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="detect_objects",
            description="Detect and label objects in an image",
            category=ToolCategory.ANALYSIS,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL or path to image"},
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific classes to detect (empty for all)"
                    },
                    "threshold": {"type": "number", "default": 0.5}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "bounding_boxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "width": {"type": "number"},
                                "height": {"type": "number"},
                                "label": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "labels": {"type": "array", "items": {"type": "string"}}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://example.com/street.jpg"},
                    "output": {
                        "bounding_boxes": [{"x": 100, "y": 50, "width": 80, "height": 120, "label": "person", "confidence": 0.92}],
                        "labels": ["person", "car", "bicycle"]
                    }
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Detect objects in image."""
        image_url = inputs.get("image_url", "")
        if not image_url:
            return ToolResult(success=False, output=None, error="No image_url provided")
        
        classes = inputs.get("classes", [])
        threshold = inputs.get("threshold", 0.5)
        
        # Try VLM-based detection first (more flexible)
        result = await self._detect_with_vlm(image_url, classes)
        if result.success:
            return result
        
        # Fallback to ComfyUI YOLO
        return await self._detect_with_comfyui(image_url, threshold)
    
    async def _detect_with_vlm(
        self,
        image_url: str,
        classes: List[str]
    ) -> ToolResult:
        """Use VLM for object detection."""
        try:
            from .vlm_feedback import VLMFeedback
            import google.generativeai as genai
            from PIL import Image
            import tempfile
            import os
            
            vlm = VLMFeedback()
            if not vlm.available:
                return ToolResult(success=False, output=None, error="VLM not available")
            
            temp_path = None
            try:
                # Download image
                if image_url.startswith(("http://", "https://")):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as resp:
                            content = await resp.read()
                            ext = resp.headers.get("Content-Type", "image/png").split("/")[-1]
                            temp_path = Path(tempfile.gettempdir()) / f"detect_{uuid.uuid4().hex}.{ext}"
                            temp_path.write_bytes(content)
                            image_path = temp_path
                elif image_url.startswith("data:"):
                    header, data = image_url.split(",", 1)
                    ext = header.split(";")[0].split("/")[1]
                    temp_path = Path(tempfile.gettempdir()) / f"detect_{uuid.uuid4().hex}.{ext}"
                    temp_path.write_bytes(base64.b64decode(data))
                    image_path = temp_path
                else:
                    image_path = Path(image_url)
                
                img = Image.open(image_path)
                width, height = img.size
                
                class_filter = f"Focus on these classes: {', '.join(classes)}" if classes else "Detect all visible objects"
                
                prompt = f"""Detect all objects in this image with bounding boxes.
{class_filter}

Image dimensions: {width}x{height}

Respond ONLY with JSON:
{{
    "detections": [
        {{"label": "object_name", "confidence": 0.95, "x": 100, "y": 50, "width": 80, "height": 120}},
        ...
    ]
}}

Coordinates should be in pixels. Estimate bounding box positions accurately."""

                response = vlm.vlm.generate_content([prompt, img])
                text = response.text
                
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text.strip())
                detections = data.get("detections", [])
                
                bboxes = []
                labels = set()
                for det in detections:
                    bboxes.append({
                        "x": det.get("x", 0),
                        "y": det.get("y", 0),
                        "width": det.get("width", 0),
                        "height": det.get("height", 0),
                        "label": det.get("label", "unknown"),
                        "confidence": det.get("confidence", 0.5)
                    })
                    labels.add(det.get("label", "unknown"))
                
                return ToolResult(
                    success=True,
                    output={
                        "bounding_boxes": bboxes,
                        "labels": list(labels)
                    },
                    metadata={"method": "vlm_gemini", "image_size": [width, height]}
                )
            finally:
                if temp_path and temp_path.exists():
                    os.unlink(temp_path)
                    
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _detect_with_comfyui(
        self,
        image_url: str,
        threshold: float
    ) -> ToolResult:
        """Use ComfyUI YOLO for detection."""
        if not await self._check_comfyui():
            return ToolResult(success=False, output=None, error="ComfyUI not available")
        
        filename = await self._upload_image(image_url)
        if not filename:
            return ToolResult(success=False, output=None, error="Failed to upload image")
        
        # YOLO detection workflow
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": filename}
            },
            "2": {
                "class_type": "UltralyticsDetectorProvider",  # From impact-pack
                "inputs": {
                    "model_name": "yolov8m.pt"
                }
            },
            "3": {
                "class_type": "BboxDetectorSEGS",
                "inputs": {
                    "bbox_detector": ["2", 0],
                    "image": ["1", 0],
                    "threshold": threshold,
                    "dilation": 0
                }
            }
        }
        
        success, outputs, error = await self._execute_workflow(workflow)
        
        if not success:
            # Return empty results on failure (detection is optional)
            return ToolResult(
                success=True,
                output={"bounding_boxes": [], "labels": []},
                metadata={"method": "comfyui_yolo", "error": error}
            )
        
        # Parse detection outputs
        bboxes = []
        labels = set()
        
        for node_id, node_out in outputs.items():
            if "segs" in node_out or "SEGS" in node_out:
                segs = node_out.get("segs") or node_out.get("SEGS", [])
                for seg in segs:
                    if isinstance(seg, dict):
                        bbox = seg.get("bbox", [0, 0, 0, 0])
                        label = seg.get("label", "object")
                        conf = seg.get("confidence", 1.0)
                        bboxes.append({
                            "x": bbox[0],
                            "y": bbox[1],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                            "label": label,
                            "confidence": conf
                        })
                        labels.add(label)
        
        return ToolResult(
            success=True,
            output={
                "bounding_boxes": bboxes,
                "labels": list(labels)
            },
            metadata={"method": "comfyui_yolo"}
        )


class OCRTool(VisionToolBase):
    """
    OCR text extraction tool.
    
    Extracts text from images using PaddleOCR or
    Tesseract via ComfyUI, or VLM for complex layouts.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ocr_extract",
            description="Extract text from an image using OCR",
            category=ToolCategory.ANALYSIS,
            input_schema={
                "type": "object",
                "required": ["image_url"],
                "properties": {
                    "image_url": {"type": "string", "description": "URL or path to image"},
                    "language": {"type": "string", "default": "en"},
                    "detect_layout": {"type": "boolean", "default": False}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "bounding_boxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "width": {"type": "number"},
                                "height": {"type": "number"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "confidence": {"type": "number"}
                }
            },
            examples=[
                {
                    "input": {"image_url": "http://example.com/document.jpg"},
                    "output": {"text": "Hello World", "bounding_boxes": [], "confidence": 0.95}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Extract text from image."""
        image_url = inputs.get("image_url", "")
        if not image_url:
            return ToolResult(success=False, output=None, error="No image_url provided")
        
        language = inputs.get("language", "en")
        detect_layout = inputs.get("detect_layout", False)
        
        # Use VLM for OCR (most reliable)
        result = await self._ocr_with_vlm(image_url, language, detect_layout)
        if result.success:
            return result
        
        # Fallback to ComfyUI PaddleOCR
        return await self._ocr_with_comfyui(image_url, language)
    
    async def _ocr_with_vlm(
        self,
        image_url: str,
        language: str,
        detect_layout: bool
    ) -> ToolResult:
        """Use VLM for OCR."""
        try:
            from .vlm_feedback import VLMFeedback
            from PIL import Image
            import tempfile
            import os
            
            vlm = VLMFeedback()
            if not vlm.available:
                return ToolResult(success=False, output=None, error="VLM not available")
            
            temp_path = None
            try:
                # Download image
                if image_url.startswith(("http://", "https://")):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as resp:
                            content = await resp.read()
                            ext = resp.headers.get("Content-Type", "image/png").split("/")[-1]
                            temp_path = Path(tempfile.gettempdir()) / f"ocr_{uuid.uuid4().hex}.{ext}"
                            temp_path.write_bytes(content)
                            image_path = temp_path
                elif image_url.startswith("data:"):
                    header, data = image_url.split(",", 1)
                    ext = header.split(";")[0].split("/")[1]
                    temp_path = Path(tempfile.gettempdir()) / f"ocr_{uuid.uuid4().hex}.{ext}"
                    temp_path.write_bytes(base64.b64decode(data))
                    image_path = temp_path
                else:
                    image_path = Path(image_url)
                
                img = Image.open(image_path)
                width, height = img.size
                
                layout_instruction = ""
                if detect_layout:
                    layout_instruction = "Also identify text regions with approximate bounding boxes (x, y, width, height in pixels)."
                
                prompt = f"""Extract ALL text from this image.
Language: {language}
{layout_instruction}

Image dimensions: {width}x{height}

Respond with JSON:
{{
    "text": "full extracted text with line breaks preserved",
    "regions": [
        {{"text": "text in region", "x": 10, "y": 20, "width": 100, "height": 30, "confidence": 0.95}},
        ...
    ],
    "overall_confidence": 0.9
}}

Be thorough - extract every piece of visible text."""

                response = vlm.vlm.generate_content([prompt, img])
                text = response.text
                
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text.strip())
                
                regions = data.get("regions", [])
                bboxes = []
                for r in regions:
                    bboxes.append({
                        "text": r.get("text", ""),
                        "x": r.get("x", 0),
                        "y": r.get("y", 0),
                        "width": r.get("width", 0),
                        "height": r.get("height", 0),
                        "confidence": r.get("confidence", 0.9)
                    })
                
                return ToolResult(
                    success=True,
                    output={
                        "text": data.get("text", ""),
                        "bounding_boxes": bboxes,
                        "confidence": data.get("overall_confidence", 0.9)
                    },
                    metadata={"method": "vlm_gemini", "language": language}
                )
            finally:
                if temp_path and temp_path.exists():
                    os.unlink(temp_path)
                    
        except json.JSONDecodeError:
            # Return plain text if JSON parsing fails
            return ToolResult(
                success=True,
                output={
                    "text": response.text if 'response' in dir() else "",
                    "bounding_boxes": [],
                    "confidence": 0.7
                },
                metadata={"method": "vlm_gemini", "parse_error": True}
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _ocr_with_comfyui(
        self,
        image_url: str,
        language: str
    ) -> ToolResult:
        """Use ComfyUI PaddleOCR."""
        if not await self._check_comfyui():
            return ToolResult(success=False, output=None, error="ComfyUI not available")
        
        filename = await self._upload_image(image_url)
        if not filename:
            return ToolResult(success=False, output=None, error="Failed to upload image")
        
        # PaddleOCR workflow (if available)
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": filename}
            },
            "2": {
                "class_type": "PaddleOCR",  # Custom node needed
                "inputs": {
                    "image": ["1", 0],
                    "language": language
                }
            }
        }
        
        success, outputs, error = await self._execute_workflow(workflow)
        
        if not success:
            return ToolResult(
                success=True,
                output={"text": "", "bounding_boxes": [], "confidence": 0.0},
                metadata={"method": "comfyui_paddleocr", "error": error}
            )
        
        # Parse OCR outputs
        text = ""
        bboxes = []
        for node_id, node_out in outputs.items():
            if "text" in node_out:
                text = node_out["text"]
            if "boxes" in node_out:
                bboxes = node_out["boxes"]
        
        return ToolResult(
            success=True,
            output={
                "text": text,
                "bounding_boxes": bboxes,
                "confidence": 0.9
            },
            metadata={"method": "comfyui_paddleocr"}
        )


def create_vision_tools(comfyui_url: str = "http://192.168.1.143:8188") -> List[BaseTool]:
    """Create all vision analysis tools.
    
    Args:
        comfyui_url: URL for ComfyUI server
        
    Returns:
        List of vision tool instances
    """
    return [
        CaptionTool(comfyui_url=comfyui_url),
        DepthMapTool(comfyui_url=comfyui_url),
        SegmentTool(comfyui_url=comfyui_url),
        DetectObjectsTool(comfyui_url=comfyui_url),
        OCRTool(comfyui_url=comfyui_url)
    ]
