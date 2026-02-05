"""ComfyUI Workflow Templates - JSON presets for common operations.

Provides reusable workflow templates for:
- txt2img (basic image generation)
- img2img (image transformation)
- inpainting (selective editing)
- upscaling (resolution enhancement)
- video (AnimateDiff)
"""

import json
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class WorkflowTemplate:
    """A reusable workflow template."""
    name: str
    description: str
    workflow: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)  # Default values
    required_nodes: list = field(default_factory=list)  # Required custom nodes
    
    def build(self, **kwargs) -> Dict[str, Any]:
        """Build workflow with custom parameters."""
        import copy
        wf = copy.deepcopy(self.workflow)
        
        # Apply parameters to workflow nodes
        for key, value in kwargs.items():
            self._apply_param(wf, key, value)
        
        return wf
    
    def _apply_param(self, wf: Dict, param: str, value: Any) -> None:
        """Apply a parameter value to the workflow."""
        param_map = {
            'prompt': ('positive_prompt_node', 'text'),
            'negative_prompt': ('negative_prompt_node', 'text'),
            'width': ('latent_node', 'width'),
            'height': ('latent_node', 'height'),
            'steps': ('sampler_node', 'steps'),
            'cfg': ('sampler_node', 'cfg'),
            'seed': ('sampler_node', 'seed'),
            'denoise': ('sampler_node', 'denoise'),
            'checkpoint': ('checkpoint_node', 'ckpt_name'),
        }
        
        if param in param_map:
            node_key, input_key = param_map[param]
            if node_key in self.parameters:
                node_id = self.parameters[node_key]
                if node_id in wf and 'inputs' in wf[node_id]:
                    wf[node_id]['inputs'][input_key] = value


class WorkflowLibrary:
    """Library of workflow templates."""
    
    def __init__(self):
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in workflow templates."""
        self._templates['txt2img'] = self._txt2img_template()
        self._templates['img2img'] = self._img2img_template()
        self._templates['inpaint'] = self._inpaint_template()
        self._templates['upscale'] = self._upscale_template()
        self._templates['upscale_simple'] = self._upscale_simple_template()
    
    def _txt2img_template(self) -> WorkflowTemplate:
        """Basic text-to-image workflow."""
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""}
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": 512, "width": 512}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7, "denoise": 1, "latent_image": ["4", 0],
                    "model": ["1", 0], "negative": ["3", 0], "positive": ["2", 0],
                    "sampler_name": "euler", "scheduler": "normal",
                    "seed": 0, "steps": 20
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "vibe_txt2img", "images": ["6", 0]}
            }
        }
        
        return WorkflowTemplate(
            name="txt2img",
            description="Basic text-to-image generation",
            workflow=workflow,
            parameters={
                'checkpoint_node': '1',
                'positive_prompt_node': '2',
                'negative_prompt_node': '3',
                'latent_node': '4',
                'sampler_node': '5'
            }
        )
    
    def _img2img_template(self) -> WorkflowTemplate:
        """Image-to-image transformation workflow."""
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": "input.png"}
            },
            "3": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""}
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7, "denoise": 0.75, "latent_image": ["3", 0],
                    "model": ["1", 0], "negative": ["5", 0], "positive": ["4", 0],
                    "sampler_name": "euler", "scheduler": "normal",
                    "seed": 0, "steps": 20
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]}
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "vibe_img2img", "images": ["7", 0]}
            }
        }
        
        return WorkflowTemplate(
            name="img2img",
            description="Transform existing image with new prompt",
            workflow=workflow,
            parameters={
                'checkpoint_node': '1',
                'input_image_node': '2',
                'positive_prompt_node': '4',
                'negative_prompt_node': '5',
                'sampler_node': '6'
            }
        )
    
    def _inpaint_template(self) -> WorkflowTemplate:
        """Inpainting workflow for selective editing."""
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": "input.png"}
            },
            "3": {
                "class_type": "LoadImage",
                "inputs": {"image": "mask.png"}
            },
            "4": {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["2", 0],
                    "vae": ["1", 2],
                    "mask": ["3", 0],
                    "grow_mask_by": 6
                }
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""}
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7, "denoise": 1.0, "latent_image": ["4", 0],
                    "model": ["1", 0], "negative": ["6", 0], "positive": ["5", 0],
                    "sampler_name": "euler", "scheduler": "normal",
                    "seed": 0, "steps": 20
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["1", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "vibe_inpaint", "images": ["8", 0]}
            }
        }
        
        return WorkflowTemplate(
            name="inpaint",
            description="Selectively edit parts of an image using a mask",
            workflow=workflow,
            parameters={
                'checkpoint_node': '1',
                'input_image_node': '2',
                'mask_image_node': '3',
                'positive_prompt_node': '5',
                'negative_prompt_node': '6',
                'sampler_node': '7'
            }
        )
    
    def _upscale_template(self) -> WorkflowTemplate:
        """AI upscaling workflow using latent upscale."""
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": "input.png"}
            },
            "3": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}
            },
            "4": {
                "class_type": "LatentUpscale",
                "inputs": {
                    "samples": ["3", 0],
                    "upscale_method": "nearest-exact",
                    "width": 1024,
                    "height": 1024,
                    "crop": "disabled"
                }
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "high quality, detailed"}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "blurry, low quality"}
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7, "denoise": 0.5, "latent_image": ["4", 0],
                    "model": ["1", 0], "negative": ["6", 0], "positive": ["5", 0],
                    "sampler_name": "euler", "scheduler": "normal",
                    "seed": 0, "steps": 15
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["1", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "vibe_upscale", "images": ["8", 0]}
            }
        }
        
        return WorkflowTemplate(
            name="upscale",
            description="AI-powered upscaling with detail enhancement",
            workflow=workflow,
            parameters={
                'checkpoint_node': '1',
                'input_image_node': '2',
                'upscale_node': '4',
                'positive_prompt_node': '5',
                'negative_prompt_node': '6',
                'sampler_node': '7'
            }
        )
    
    def _upscale_simple_template(self) -> WorkflowTemplate:
        """Simple model-based upscaling (faster, no diffusion)."""
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "input.png"}
            },
            "2": {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": "RealESRGAN_x4plus.pth"}
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
                "inputs": {"filename_prefix": "vibe_upscale_simple", "images": ["3", 0]}
            }
        }
        
        return WorkflowTemplate(
            name="upscale_simple",
            description="Fast model-based upscaling (4x, no diffusion)",
            workflow=workflow,
            parameters={
                'input_image_node': '1',
                'upscale_model_node': '2'
            },
            required_nodes=[]  # Uses built-in nodes
        )
    
    def get(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> list:
        """List all available templates."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._templates.values()
        ]
    
    def register(self, template: WorkflowTemplate) -> None:
        """Register a custom template."""
        self._templates[template.name] = template
    
    def save_template(self, name: str, path: str) -> None:
        """Save a template to a JSON file."""
        template = self._templates.get(name)
        if template:
            with open(path, 'w') as f:
                json.dump({
                    'name': template.name,
                    'description': template.description,
                    'workflow': template.workflow,
                    'parameters': template.parameters
                }, f, indent=2)
    
    def load_template(self, path: str) -> WorkflowTemplate:
        """Load a template from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        template = WorkflowTemplate(
            name=data['name'],
            description=data.get('description', ''),
            workflow=data['workflow'],
            parameters=data.get('parameters', {})
        )
        self._templates[template.name] = template
        return template


# Convenience function
def create_workflow_library() -> WorkflowLibrary:
    """Create a workflow library with built-in templates."""
    return WorkflowLibrary()
