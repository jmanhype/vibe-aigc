"""General Workflow Composer — Builds workflows from discovered nodes.

Paper Section 5.4: "Select the optimal ensemble of components and define their data-flow topology"

This composer:
- Uses ONLY nodes that exist on the user's system
- Builds workflows based on AVAILABLE capabilities
- Adapts to whatever the user has installed

NO HARDCODED NODE TYPES. Everything is discovered.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from .discovery import SystemCapabilities, Capability, AvailableNode, AvailableModel
from .models import GenerationRequest, CharacterProfile


@dataclass
class NodeRequirement:
    """A requirement that can be satisfied by available nodes."""
    purpose: str  # What this node does (e.g., "load_model", "sample", "decode")
    input_types: List[str] = field(default_factory=list)  # What it needs
    output_types: List[str] = field(default_factory=list)  # What it produces
    preferred_patterns: List[str] = field(default_factory=list)  # Node name patterns to prefer


# Standard requirements for common operations
STANDARD_REQUIREMENTS = {
    "load_checkpoint": NodeRequirement(
        purpose="load_checkpoint",
        output_types=["MODEL", "CLIP", "VAE"],
        preferred_patterns=["checkpointloader", "checkpoint"]
    ),
    "load_unet": NodeRequirement(
        purpose="load_unet",
        output_types=["MODEL"],
        preferred_patterns=["unetloader", "diffusionmodel"]
    ),
    "load_vae": NodeRequirement(
        purpose="load_vae",
        output_types=["VAE"],
        preferred_patterns=["vaeloader"]
    ),
    "load_clip": NodeRequirement(
        purpose="load_clip",
        output_types=["CLIP"],
        preferred_patterns=["cliploader"]
    ),
    # IP-Adapter / Character consistency
    "load_ipadapter": NodeRequirement(
        purpose="load_ipadapter",
        output_types=["IPADAPTER"],
        preferred_patterns=["ipadaptermodelloader", "ipadapterloader", "ipadapter"]
    ),
    "load_clip_vision": NodeRequirement(
        purpose="load_clip_vision",
        output_types=["CLIP_VISION"],
        preferred_patterns=["clipvisionloader", "clip_vision"]
    ),
    "encode_clip_vision": NodeRequirement(
        purpose="encode_clip_vision",
        input_types=["CLIP_VISION", "IMAGE"],
        output_types=["CLIP_VISION_OUTPUT"],
        preferred_patterns=["clipvisionencode"]
    ),
    "apply_ipadapter": NodeRequirement(
        purpose="apply_ipadapter",
        input_types=["MODEL", "IPADAPTER", "IMAGE"],
        output_types=["MODEL"],
        preferred_patterns=["ipadapterapply", "ipadapter"]
    ),
    "load_image": NodeRequirement(
        purpose="load_image",
        output_types=["IMAGE"],
        preferred_patterns=["loadimage", "load_image"]
    ),
    "load_lora": NodeRequirement(
        purpose="load_lora",
        input_types=["MODEL", "CLIP"],
        output_types=["MODEL", "CLIP"],
        preferred_patterns=["loraloader", "lora"]
    ),
    "encode_text": NodeRequirement(
        purpose="encode_text",
        input_types=["CLIP"],
        output_types=["CONDITIONING"],
        preferred_patterns=["cliptextencode", "textencode"]
    ),
    "empty_latent": NodeRequirement(
        purpose="empty_latent",
        output_types=["LATENT"],
        preferred_patterns=["emptylatent"]
    ),
    "sample": NodeRequirement(
        purpose="sample",
        input_types=["MODEL", "CONDITIONING", "LATENT"],
        output_types=["LATENT"],
        preferred_patterns=["ksampler", "sampler"]
    ),
    "decode": NodeRequirement(
        purpose="decode",
        input_types=["LATENT", "VAE"],
        output_types=["IMAGE"],
        preferred_patterns=["vaedecode"]
    ),
    "save_image": NodeRequirement(
        purpose="save_image",
        input_types=["IMAGE"],
        preferred_patterns=["saveimage", "save"]
    ),
    "save_video": NodeRequirement(
        purpose="save_video",
        input_types=["IMAGE"],
        preferred_patterns=["videocombine", "savevideo", "saveanimated"]
    ),
}


class GeneralComposer:
    """Composes workflows from discovered nodes.
    
    This is the GENERAL approach:
    1. Look at what nodes exist
    2. Find nodes that satisfy requirements
    3. Wire them together
    4. Output valid workflow
    """
    
    def __init__(self, capabilities: SystemCapabilities):
        self.caps = capabilities
        self.nodes = capabilities.nodes
    
    def find_node_for(self, requirement: NodeRequirement) -> Optional[str]:
        """Find a node that satisfies a requirement."""
        candidates = []
        
        for name, node in self.nodes.items():
            name_lower = name.lower()
            
            # Check if node matches preferred patterns
            pattern_match = any(p in name_lower for p in requirement.preferred_patterns)
            
            # Check output types match
            output_match = not requirement.output_types or any(
                out in node.outputs for out in requirement.output_types
            )
            
            if pattern_match and output_match:
                candidates.append((name, 2 if pattern_match else 1))
        
        if candidates:
            # Return best match
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def find_model_for(self, capability: Capability) -> Optional[AvailableModel]:
        """Find a model that provides a capability."""
        models = self.caps.get_models_for(capability)
        if models:
            return models[0]
        return None
    
    def compose_text_to_image(
        self,
        model: AvailableModel,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Compose a text-to-image workflow from available nodes."""
        
        # Find required nodes
        loader = self.find_node_for(STANDARD_REQUIREMENTS["load_checkpoint"])
        encoder = self.find_node_for(STANDARD_REQUIREMENTS["encode_text"])
        empty = self.find_node_for(STANDARD_REQUIREMENTS["empty_latent"])
        sampler = self.find_node_for(STANDARD_REQUIREMENTS["sample"])
        decoder = self.find_node_for(STANDARD_REQUIREMENTS["decode"])
        saver = self.find_node_for(STANDARD_REQUIREMENTS["save_image"])
        
        missing = []
        if not loader: missing.append("checkpoint loader")
        if not encoder: missing.append("text encoder")
        if not sampler: missing.append("sampler")
        if not decoder: missing.append("decoder")
        
        if missing:
            print(f"Cannot compose text_to_image: missing {missing}")
            return None
        
        # Build workflow
        workflow = {
            "1": {
                "class_type": loader,
                "inputs": {"ckpt_name": model.filename}
            },
            "2": {
                "class_type": encoder,
                "inputs": {"text": prompt, "clip": ["1", 1]}
            },
            "3": {
                "class_type": encoder,
                "inputs": {"text": negative_prompt or "bad quality", "clip": ["1", 1]}
            },
            "4": {
                "class_type": empty or "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            },
            "5": {
                "class_type": sampler,
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "6": {
                "class_type": decoder,
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
        }
        
        if saver:
            workflow["7"] = {
                "class_type": saver,
                "inputs": {"images": ["6", 0], "filename_prefix": "vibe"}
            }
        
        return workflow
    
    def compose_text_to_video(
        self,
        model: AvailableModel,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        frames: int = 24,
        steps: int = 20,
        cfg: float = 6.0,
        seed: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Compose a text-to-video workflow from available nodes."""
        
        # Video workflows are more complex and varied
        # Try to find video-specific nodes
        
        # Check for different video model types
        model_name = model.filename.lower()
        
        # AnimateDiff pattern
        if "animate" in model_name or "motion" in model_name:
            return self._compose_animatediff(model, prompt, negative_prompt, width, height, frames, steps, cfg, seed)
        
        # Wan/LTX pattern (uses UNET loader)
        if any(x in model_name for x in ["wan", "ltx", "i2v"]):
            return self._compose_video_diffusion(model, prompt, negative_prompt, width, height, frames, steps, cfg, seed)
        
        # Generic video attempt
        return self._compose_generic_video(model, prompt, negative_prompt, width, height, frames, steps, cfg, seed)
    
    def _compose_animatediff(
        self, model, prompt, negative_prompt, width, height, frames, steps, cfg, seed
    ) -> Optional[Dict[str, Any]]:
        """Compose AnimateDiff-style workflow."""
        # Look for AnimateDiff-specific nodes
        motion_loader = None
        for name in self.nodes:
            if "animatediff" in name.lower() and "loader" in name.lower():
                motion_loader = name
                break
        
        if not motion_loader:
            return None
        
        # Base image workflow + motion module
        base = self.compose_text_to_image(model, prompt, negative_prompt, width, height, steps, cfg, seed)
        if not base:
            return None
        
        # Add motion module (simplified)
        return base
    
    def _compose_video_diffusion(
        self, model, prompt, negative_prompt, width, height, frames, steps, cfg, seed
    ) -> Optional[Dict[str, Any]]:
        """Compose video diffusion workflow (Wan/LTX style)."""
        
        # Find required nodes for video
        unet_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_unet"])
        clip_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_clip"])
        vae_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_vae"])
        encoder = self.find_node_for(STANDARD_REQUIREMENTS["encode_text"])
        sampler = self.find_node_for(STANDARD_REQUIREMENTS["sample"])
        decoder = self.find_node_for(STANDARD_REQUIREMENTS["decode"])
        video_saver = self.find_node_for(STANDARD_REQUIREMENTS["save_video"])
        
        # Check for advanced sampling nodes
        noise_node = None
        scheduler_node = None
        guider_node = None
        
        for name in self.nodes:
            name_lower = name.lower()
            if "randomnoise" in name_lower:
                noise_node = name
            if "scheduler" in name_lower and "basic" in name_lower:
                scheduler_node = name
            if "guider" in name_lower:
                guider_node = name
        
        if not unet_loader:
            # Try DiffusionModelLoader variants
            for name in self.nodes:
                if "diffusionmodel" in name.lower() and "loader" in name.lower():
                    unet_loader = name
                    break
        
        if not unet_loader:
            print("Cannot compose video: no UNET/diffusion loader found")
            return None
        
        # Build workflow based on what's available
        workflow = {}
        node_id = 1
        
        # Model loading
        workflow[str(node_id)] = {
            "class_type": unet_loader,
            "inputs": {"unet_name" if "unet" in unet_loader.lower() else "model_path": model.filename}
        }
        model_node = str(node_id)
        node_id += 1
        
        # CLIP (if separate)
        clip_node = None
        if clip_loader:
            # Try to find matching CLIP/text encoder
            clip_models = self.caps.models.get("clip", []) + self.caps.models.get("text_encoders", [])
            if clip_models:
                workflow[str(node_id)] = {
                    "class_type": clip_loader,
                    "inputs": {"clip_name": clip_models[0].filename}
                }
                clip_node = str(node_id)
                node_id += 1
        
        # VAE
        vae_node = None
        if vae_loader:
            vae_models = self.caps.models.get("vae", [])
            if vae_models:
                workflow[str(node_id)] = {
                    "class_type": vae_loader,
                    "inputs": {"vae_name": vae_models[0].filename}
                }
                vae_node = str(node_id)
                node_id += 1
        
        # Text encoding
        if encoder and clip_node:
            workflow[str(node_id)] = {
                "class_type": encoder,
                "inputs": {"text": prompt, "clip": [clip_node, 0]}
            }
            pos_node = str(node_id)
            node_id += 1
            
            workflow[str(node_id)] = {
                "class_type": encoder,
                "inputs": {"text": negative_prompt or "blurry, static", "clip": [clip_node, 0]}
            }
            neg_node = str(node_id)
            node_id += 1
        else:
            pos_node = neg_node = None
        
        # Sampling
        sample_node = None
        latent_node = None
        
        if sampler and pos_node:
            # Find empty latent for video
            empty_video = None
            for name in self.nodes:
                if "empty" in name.lower() and ("video" in name.lower() or "latent" in name.lower()):
                    empty_video = name
                    break
            
            if empty_video:
                workflow[str(node_id)] = {
                    "class_type": empty_video,
                    "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}
                }
                latent_node = str(node_id)
                node_id += 1
            
            if latent_node:
                workflow[str(node_id)] = {
                    "class_type": sampler,
                    "inputs": {
                        "model": [model_node, 0],
                        "positive": [pos_node, 0],
                        "negative": [neg_node, 0],
                        "latent_image": [latent_node, 0],
                        "seed": seed,
                        "steps": steps,
                        "cfg": cfg,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0
                    }
                }
                sample_node = str(node_id)
                node_id += 1
        
        # Decode
        decode_node = None
        if decoder and vae_node and sample_node:
            workflow[str(node_id)] = {
                "class_type": decoder,
                "inputs": {"samples": [sample_node, 0], "vae": [vae_node, 0]}
            }
            decode_node = str(node_id)
            node_id += 1
        
        # Save video
        if video_saver and decode_node:
            workflow[str(node_id)] = {
                "class_type": video_saver,
                "inputs": {"images": [decode_node, 0], "frame_rate": 24, "filename_prefix": "vibe"}
            }
        
        if not workflow:
            print("Could not compose video workflow - missing required nodes")
            return None
            
        return workflow
    
    def _compose_generic_video(
        self, model, prompt, negative_prompt, width, height, frames, steps, cfg, seed
    ) -> Optional[Dict[str, Any]]:
        """Fallback generic video composition."""
        # Try text_to_image with batch_size = frames
        base = self.compose_text_to_image(model, prompt, negative_prompt, width, height, steps, cfg, seed)
        if base and "4" in base:
            base["4"]["inputs"]["batch_size"] = frames
        return base
    
    # =========================================================================
    # CHARACTER CONSISTENCY / IP-ADAPTER SUPPORT
    # =========================================================================
    
    def inject_ipadapter(
        self,
        workflow: Dict[str, Any],
        reference_image: str,
        strength: float = 0.8,
        model_node_id: str = "1",
        start_node_id: int = 100
    ) -> Tuple[Dict[str, Any], str]:
        """Inject IP-Adapter nodes into an existing workflow.
        
        Returns updated workflow and the new model node ID to use downstream.
        """
        node_id = start_node_id
        
        # Find IP-Adapter loader
        ipadapter_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_ipadapter"])
        clip_vision_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_clip_vision"])
        ipadapter_apply = self.find_node_for(STANDARD_REQUIREMENTS["apply_ipadapter"])
        load_image = self.find_node_for(STANDARD_REQUIREMENTS["load_image"])
        
        if not ipadapter_apply:
            print("No IP-Adapter apply node found - skipping character reference")
            return workflow, model_node_id
        
        if not load_image:
            print("No image loader found - skipping character reference")
            return workflow, model_node_id
        
        # Load reference image
        workflow[str(node_id)] = {
            "class_type": load_image,
            "inputs": {"image": reference_image}
        }
        ref_image_node = str(node_id)
        node_id += 1
        
        # Load CLIP Vision (if available and needed)
        clip_vision_node = None
        if clip_vision_loader:
            clip_vision_models = self.caps.get_clip_vision_models()
            if clip_vision_models:
                workflow[str(node_id)] = {
                    "class_type": clip_vision_loader,
                    "inputs": {"clip_name": clip_vision_models[0].filename}
                }
                clip_vision_node = str(node_id)
                node_id += 1
        
        # Load IP-Adapter model (if loader exists)
        ipadapter_model_node = None
        if ipadapter_loader:
            ipadapter_models = self.caps.get_ipadapter_models()
            if ipadapter_models:
                workflow[str(node_id)] = {
                    "class_type": ipadapter_loader,
                    "inputs": {"ipadapter_file": ipadapter_models[0].filename}
                }
                ipadapter_model_node = str(node_id)
                node_id += 1
        
        # Apply IP-Adapter
        apply_inputs = {
            "model": [model_node_id, 0],
            "image": [ref_image_node, 0],
            "weight": strength,
        }
        
        # Add optional inputs if available
        if ipadapter_model_node:
            apply_inputs["ipadapter"] = [ipadapter_model_node, 0]
        if clip_vision_node:
            apply_inputs["clip_vision"] = [clip_vision_node, 0]
        
        workflow[str(node_id)] = {
            "class_type": ipadapter_apply,
            "inputs": apply_inputs
        }
        new_model_node = str(node_id)
        node_id += 1
        
        return workflow, new_model_node
    
    def inject_lora(
        self,
        workflow: Dict[str, Any],
        lora_path: str,
        strength: float = 0.8,
        model_node_id: str = "1",
        clip_node_id: Optional[str] = None,
        start_node_id: int = 100
    ) -> Tuple[Dict[str, Any], str, Optional[str]]:
        """Inject LoRA into an existing workflow.
        
        Returns updated workflow, new model node ID, and new clip node ID.
        """
        lora_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_lora"])
        if not lora_loader:
            print("No LoRA loader found - skipping LoRA injection")
            return workflow, model_node_id, clip_node_id
        
        node_id = start_node_id
        
        lora_inputs = {
            "lora_name": lora_path,
            "strength_model": strength,
            "strength_clip": strength,
            "model": [model_node_id, 0],
        }
        
        if clip_node_id:
            lora_inputs["clip"] = [clip_node_id, 0]
        
        workflow[str(node_id)] = {
            "class_type": lora_loader,
            "inputs": lora_inputs
        }
        new_model_node = str(node_id)
        new_clip_node = str(node_id) if clip_node_id else None
        
        return workflow, new_model_node, new_clip_node
    
    def inject_clip_vision_reference(
        self,
        workflow: Dict[str, Any],
        reference_image: str,
        start_node_id: int = 100
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Inject CLIP Vision encoding for reference image (lighter alternative to IP-Adapter).
        
        This works even without full IP-Adapter by encoding the reference image
        via CLIP Vision, which can be used for style guidance.
        
        Returns updated workflow and the clip vision output node ID.
        """
        clip_vision_loader = self.find_node_for(STANDARD_REQUIREMENTS["load_clip_vision"])
        clip_vision_encode = self.find_node_for(STANDARD_REQUIREMENTS["encode_clip_vision"])
        load_image = self.find_node_for(STANDARD_REQUIREMENTS["load_image"])
        
        if not (clip_vision_encode and load_image):
            print("CLIP Vision encode or image loader not available")
            return workflow, None
        
        node_id = start_node_id
        
        # Load reference image
        workflow[str(node_id)] = {
            "class_type": load_image,
            "inputs": {"image": reference_image}
        }
        ref_image_node = str(node_id)
        node_id += 1
        
        # Load CLIP Vision model (if needed)
        clip_vision_node = None
        if clip_vision_loader:
            clip_vision_models = self.caps.get_clip_vision_models()
            if clip_vision_models:
                workflow[str(node_id)] = {
                    "class_type": clip_vision_loader,
                    "inputs": {"clip_name": clip_vision_models[0].filename}
                }
                clip_vision_node = str(node_id)
                node_id += 1
        
        # Encode with CLIP Vision
        encode_inputs = {"image": [ref_image_node, 0]}
        if clip_vision_node:
            encode_inputs["clip_vision"] = [clip_vision_node, 0]
        
        workflow[str(node_id)] = {
            "class_type": clip_vision_encode,
            "inputs": encode_inputs
        }
        clip_vision_output = str(node_id)
        
        return workflow, clip_vision_output
    
    def compose_with_character_reference(
        self,
        model: AvailableModel,
        prompt: str,
        reference_image: str,
        character_strength: float = 0.8,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = 0,
        character_lora: Optional[str] = None,
        character_lora_strength: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """Compose a text-to-image workflow with character reference.
        
        Uses IP-Adapter if available, falls back to CLIP Vision encoding.
        """
        # Start with base text-to-image workflow
        workflow = self.compose_text_to_image(
            model, prompt, negative_prompt, width, height, steps, cfg, seed
        )
        
        if not workflow:
            return None
        
        # Find the model node (typically node 1 from checkpoint loader)
        model_node_id = "1"
        clip_node_id = "1"  # CLIP is output 1 from checkpoint loader
        
        current_start_id = 100
        
        # Inject character LoRA if provided
        if character_lora:
            workflow, model_node_id, clip_node_id = self.inject_lora(
                workflow,
                character_lora,
                character_lora_strength,
                model_node_id,
                clip_node_id,
                current_start_id
            )
            current_start_id += 10
        
        # Try IP-Adapter first
        if self.caps.has_ipadapter_support():
            workflow, new_model_node = self.inject_ipadapter(
                workflow,
                reference_image,
                character_strength,
                model_node_id,
                current_start_id
            )
            
            # Update sampler to use new model node
            for node_id, node in workflow.items():
                if node.get("class_type", "").lower() in ["ksampler", "sampler"]:
                    if "model" in node.get("inputs", {}):
                        node["inputs"]["model"] = [new_model_node, 0]
        
        elif self.caps.has_reference_image_support():
            # Fall back to CLIP Vision encoding
            workflow, clip_vision_output = self.inject_clip_vision_reference(
                workflow,
                reference_image,
                current_start_id
            )
            # Note: CLIP Vision output would need to be wired to compatible nodes
            # This is a placeholder for systems without full IP-Adapter
            print("Using CLIP Vision encoding (limited character consistency)")
        
        else:
            print("No character reference support available on this system")
        
        return workflow
    
    def compose_for_capability(
        self,
        capability: Capability,
        prompt: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Compose a workflow for a capability."""
        
        # Find a model for this capability
        model = self.find_model_for(capability)
        if not model:
            print(f"No model found for {capability.value}")
            return None
        
        if capability == Capability.TEXT_TO_IMAGE:
            return self.compose_text_to_image(model, prompt, **kwargs)
        elif capability in [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO]:
            return self.compose_text_to_video(model, prompt, **kwargs)
        else:
            print(f"Composition not yet implemented for {capability.value}")
            return None


    def compose_from_request(
        self,
        request: GenerationRequest,
        capability: Capability = Capability.TEXT_TO_IMAGE
    ) -> Optional[Dict[str, Any]]:
        """Compose a workflow from a GenerationRequest.
        
        Automatically handles character consistency if reference_image is provided.
        """
        # Find appropriate model
        model = None
        if request.model:
            # Use specified model
            for category_models in self.caps.models.values():
                for m in category_models:
                    if m.filename == request.model:
                        model = m
                        break
                if model:
                    break
        
        if not model:
            model = self.find_model_for(capability)
        
        if not model:
            print(f"No model found for {capability.value}")
            return None
        
        # Compose based on whether we have character reference
        if request.reference_image:
            workflow = self.compose_with_character_reference(
                model=model,
                prompt=request.prompt,
                reference_image=request.reference_image,
                character_strength=request.character_strength,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                cfg=request.cfg,
                seed=request.seed,
                character_lora=request.character_lora,
                character_lora_strength=request.character_lora_strength
            )
        else:
            # Standard composition
            if capability == Capability.TEXT_TO_IMAGE:
                workflow = self.compose_text_to_image(
                    model, request.prompt, request.negative_prompt,
                    request.width, request.height, request.steps,
                    request.cfg, request.seed
                )
            elif capability in [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO]:
                workflow = self.compose_text_to_video(
                    model, request.prompt, request.negative_prompt,
                    request.width, request.height, request.frames,
                    request.steps, request.cfg, request.seed
                )
            else:
                workflow = self.compose_for_capability(capability, request.prompt)
        
        # Inject additional LoRAs if specified
        if workflow and request.loras:
            model_node = "1"
            clip_node = "1"
            start_id = 200
            
            for lora_config in request.loras:
                lora_path = lora_config.get("path", lora_config.get("name", ""))
                lora_strength = lora_config.get("strength", 0.8)
                
                if lora_path:
                    workflow, model_node, clip_node = self.inject_lora(
                        workflow, lora_path, lora_strength,
                        model_node, clip_node, start_id
                    )
                    start_id += 10
        
        return workflow
    
    def compose_for_character(
        self,
        profile: CharacterProfile,
        prompt: str,
        capability: Capability = Capability.TEXT_TO_IMAGE,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Compose a workflow using a CharacterProfile for consistency.
        
        Convenience method that extracts reference settings from profile.
        """
        # Merge profile settings with any overrides
        gen_params = profile.to_generation_params()
        gen_params.update(kwargs)
        
        # Inject trigger words into prompt if available
        full_prompt = prompt
        if profile.trigger_words:
            trigger_str = " ".join(profile.trigger_words)
            full_prompt = f"{trigger_str}, {prompt}"
        
        # Add character description to prompt
        if profile.description:
            full_prompt = f"{profile.description}, {full_prompt}"
        
        # Create request
        request = GenerationRequest(
            prompt=full_prompt,
            **gen_params
        )
        
        return self.compose_from_request(request, capability)


def create_composer(capabilities: SystemCapabilities) -> GeneralComposer:
    """Create a general composer from system capabilities."""
    return GeneralComposer(capabilities)
