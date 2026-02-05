"""
Model Registry - Auto-detect and manage available models.

This is what was MISSING from our implementation.
The system should KNOW what it can do without being told.

Features:
- Auto-detect models installed in ComfyUI
- Categorize by capability (image, video, audio)
- Track model specs (VRAM, quality, speed)
- Research new models via Perplexity
- Recommend best model for a task
"""

import asyncio
import aiohttp
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import os


class ModelCapability(Enum):
    """What a model can do."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    TEXT_TO_AUDIO = "text_to_audio"
    AUDIO_TO_AUDIO = "audio_to_audio"
    UPSCALE = "upscale"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"


class ModelFamily(Enum):
    """Model architecture families."""
    SD15 = "sd1.5"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    LTX_VIDEO = "ltx_video"
    ANIMATEDIFF = "animatediff"
    HUNYUAN = "hunyuan"
    MOCHI = "mochi"
    MUSICGEN = "musicgen"
    AUDIOLDM = "audioldm"
    UNKNOWN = "unknown"


@dataclass
class ModelSpec:
    """Specification for a model."""
    name: str
    filename: str
    family: ModelFamily
    capabilities: List[ModelCapability]
    vram_required: float  # GB
    quality_tier: int  # 1-10
    speed_tier: int  # 1-10 (10 = fastest)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "family": self.family.value,
            "capabilities": [c.value for c in self.capabilities],
            "vram_required": self.vram_required,
            "quality_tier": self.quality_tier,
            "speed_tier": self.speed_tier,
            "notes": self.notes
        }


# Known model patterns and their specs
KNOWN_MODELS = {
    # LTX Video
    "ltxv": {
        "family": ModelFamily.LTX_VIDEO,
        "capabilities": [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO],
        "patterns": ["ltxv", "ltx-video", "ltx_video"],
    },
    # AnimateDiff
    "animatediff": {
        "family": ModelFamily.ANIMATEDIFF,
        "capabilities": [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO],
        "patterns": ["animatediff", "mm_sd", "motion_module", "v3_adapter"],
    },
    # SDXL
    "sdxl": {
        "family": ModelFamily.SDXL,
        "capabilities": [ModelCapability.TEXT_TO_IMAGE, ModelCapability.IMAGE_TO_IMAGE],
        "patterns": ["sdxl", "sd_xl", "juggernaut", "realvis"],
    },
    # SD 1.5
    "sd15": {
        "family": ModelFamily.SD15,
        "capabilities": [ModelCapability.TEXT_TO_IMAGE, ModelCapability.IMAGE_TO_IMAGE],
        "patterns": ["v1-5", "dreamshaper", "realistic", "deliberate", "revanimated"],
    },
    # Flux
    "flux": {
        "family": ModelFamily.FLUX,
        "capabilities": [ModelCapability.TEXT_TO_IMAGE],
        "patterns": ["flux"],
    },
}


class ModelRegistry:
    """
    Dynamic registry of available models.
    
    Auto-detects what's installed in ComfyUI and categorizes
    by capability so the MetaPlanner can make informed decisions.
    """
    
    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        perplexity_key: Optional[str] = None
    ):
        self.comfyui_url = comfyui_url
        self.perplexity_key = perplexity_key or os.environ.get("PERPLEXITY_API_KEY")
        self.models: Dict[str, ModelSpec] = {}
        self.capabilities: Dict[ModelCapability, List[ModelSpec]] = {
            cap: [] for cap in ModelCapability
        }
        self._object_info: Dict = {}
    
    async def refresh(self) -> None:
        """Refresh the model registry from ComfyUI."""
        async with aiohttp.ClientSession() as session:
            # Get all available models from ComfyUI
            async with session.get(f"{self.comfyui_url}/object_info") as resp:
                self._object_info = await resp.json()
        
        # Extract models from various loader nodes
        await self._scan_checkpoints()
        await self._scan_unets()
        await self._scan_loras()
        await self._scan_vaes()
        await self._scan_motion_modules()
        
        # Categorize by capability
        self._categorize_models()
    
    async def _scan_checkpoints(self) -> None:
        """Scan checkpoint models."""
        node = self._object_info.get("CheckpointLoaderSimple", {})
        ckpts = node.get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
        
        for ckpt in (ckpts or []):
            spec = self._identify_model(ckpt, "checkpoint")
            if spec:
                self.models[ckpt] = spec
    
    async def _scan_unets(self) -> None:
        """Scan UNET models (for video models like LTX)."""
        node = self._object_info.get("UNETLoader", {})
        unets = node.get("input", {}).get("required", {}).get("unet_name", [[]])[0]
        
        for unet in (unets or []):
            spec = self._identify_model(unet, "unet")
            if spec:
                self.models[unet] = spec
    
    async def _scan_loras(self) -> None:
        """Scan LoRA models."""
        node = self._object_info.get("LoraLoader", {})
        loras = node.get("input", {}).get("required", {}).get("lora_name", [[]])[0]
        
        for lora in (loras or []):
            # LoRAs modify existing models, track separately
            pass
    
    async def _scan_vaes(self) -> None:
        """Scan VAE models."""
        node = self._object_info.get("VAELoader", {})
        vaes = node.get("input", {}).get("required", {}).get("vae_name", [[]])[0]
        # VAEs are support models, not primary generators
    
    async def _scan_motion_modules(self) -> None:
        """Scan AnimateDiff motion modules."""
        node = self._object_info.get("ADE_AnimateDiffLoaderWithContext", {})
        if node:
            modules = node.get("input", {}).get("required", {}).get("model_name", [[]])[0]
            for mod in (modules or []):
                spec = self._identify_model(mod, "motion_module")
                if spec:
                    self.models[mod] = spec
    
    def _identify_model(self, filename: str, model_type: str) -> Optional[ModelSpec]:
        """Identify a model from its filename."""
        lower = filename.lower()
        
        for key, info in KNOWN_MODELS.items():
            for pattern in info["patterns"]:
                if pattern in lower:
                    # Estimate VRAM based on file patterns
                    vram = self._estimate_vram(filename)
                    quality = self._estimate_quality(filename)
                    speed = self._estimate_speed(filename, vram)
                    
                    return ModelSpec(
                        name=self._clean_name(filename),
                        filename=filename,
                        family=info["family"],
                        capabilities=info["capabilities"],
                        vram_required=vram,
                        quality_tier=quality,
                        speed_tier=speed,
                        notes=f"Auto-detected from {model_type}"
                    )
        
        return None
    
    def _estimate_vram(self, filename: str) -> float:
        """Estimate VRAM requirement from filename."""
        lower = filename.lower()
        
        if "fp8" in lower or "q4" in lower or "q8" in lower:
            return 4.0  # Quantized
        if "fp16" in lower:
            return 6.0
        if "xl" in lower or "sdxl" in lower:
            return 8.0
        if "ltxv" in lower and "2b" in lower:
            return 6.0
        if "13b" in lower:
            return 12.0
        
        return 4.0  # Default for SD1.5
    
    def _estimate_quality(self, filename: str) -> int:
        """Estimate quality tier from filename."""
        lower = filename.lower()
        
        if "distilled" in lower:
            return 7  # Slightly lower quality for speed
        if "turbo" in lower or "lcm" in lower:
            return 6  # Speed optimized
        if "xl" in lower:
            return 8
        if "ltxv" in lower:
            return 9  # State of the art video
        
        return 7  # Default
    
    def _estimate_speed(self, filename: str, vram: float) -> int:
        """Estimate speed tier."""
        lower = filename.lower()
        
        if "distilled" in lower or "turbo" in lower:
            return 9  # Fast
        if "lcm" in lower:
            return 10  # Very fast
        if vram > 8:
            return 4  # Slow due to size
        
        return 6  # Default
    
    def _clean_name(self, filename: str) -> str:
        """Clean filename to readable name."""
        name = filename.replace(".safetensors", "").replace(".ckpt", "")
        name = name.replace("_", " ").replace("-", " ")
        return name.title()
    
    def _categorize_models(self) -> None:
        """Categorize models by capability."""
        self.capabilities = {cap: [] for cap in ModelCapability}
        
        for spec in self.models.values():
            for cap in spec.capabilities:
                self.capabilities[cap].append(spec)
    
    def get_best_for(
        self,
        capability: ModelCapability,
        max_vram: float = 8.0,
        prefer_quality: bool = True
    ) -> Optional[ModelSpec]:
        """Get the best model for a capability given constraints."""
        candidates = [
            m for m in self.capabilities.get(capability, [])
            if m.vram_required <= max_vram
        ]
        
        if not candidates:
            return None
        
        # Sort by quality or speed
        if prefer_quality:
            candidates.sort(key=lambda m: m.quality_tier, reverse=True)
        else:
            candidates.sort(key=lambda m: m.speed_tier, reverse=True)
        
        return candidates[0]
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get all available capabilities."""
        return [
            cap for cap, models in self.capabilities.items()
            if models
        ]
    
    async def research_model(self, query: str) -> str:
        """Research a model type using Perplexity."""
        if not self.perplexity_key:
            return "No Perplexity API key configured"
        
        import urllib.request
        
        data = json.dumps({
            "model": "sonar",
            "messages": [{
                "role": "user",
                "content": f"What is the best open-source AI model for: {query}? "
                          f"Focus on models that work with ComfyUI. "
                          f"Give the Hugging Face link and VRAM requirements."
            }]
        }).encode()
        
        req = urllib.request.Request(
            "https://api.perplexity.ai/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.perplexity_key}",
                "Content-Type": "application/json"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Research failed: {e}"
    
    def summary(self) -> str:
        """Get a summary of available capabilities."""
        lines = ["=== Model Registry ===", ""]
        
        for cap in ModelCapability:
            models = self.capabilities.get(cap, [])
            if models:
                lines.append(f"{cap.value}:")
                for m in models[:3]:  # Top 3
                    lines.append(f"  - {m.name} ({m.vram_required}GB, Q{m.quality_tier})")
                if len(models) > 3:
                    lines.append(f"  ... and {len(models)-3} more")
                lines.append("")
        
        return "\n".join(lines)


async def demo():
    """Demo the model registry."""
    registry = ModelRegistry()
    
    print("Scanning ComfyUI for available models...")
    await registry.refresh()
    
    print(registry.summary())
    
    # Get best video model
    video_model = registry.get_best_for(ModelCapability.TEXT_TO_VIDEO, max_vram=8.0)
    if video_model:
        print(f"\nBest video model for 8GB VRAM: {video_model.name}")
        print(f"  File: {video_model.filename}")
        print(f"  Quality: {video_model.quality_tier}/10")


if __name__ == "__main__":
    asyncio.run(demo())
