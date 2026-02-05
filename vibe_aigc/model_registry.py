"""
Model Registry - Auto-detect and manage available models.

The system KNOWS what it can do without being told.

Features:
- Auto-detect hardware (GPU, VRAM)
- Auto-detect models installed in ComfyUI
- Categorize by capability (image, video, audio)
- Track model specs (VRAM, quality, speed)
- Research new models via Perplexity
- Download missing models via Comfy-Pilot
- Recommend best model for a task
- Auto-upgrade to better models
"""

import asyncio
import aiohttp
import json
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import os
import sys


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


@dataclass
class HardwareSpec:
    """Hardware specifications for model compatibility."""
    gpu_name: str = "Unknown"
    vram_gb: float = 8.0
    ram_gb: float = 32.0
    
    @classmethod
    def detect(cls) -> "HardwareSpec":
        """Auto-detect hardware specs."""
        try:
            import subprocess
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    return cls(
                        gpu_name=parts[0],
                        vram_gb=float(parts[1]) / 1024,  # MB to GB
                        ram_gb=32.0  # Default
                    )
        except:
            pass
        return cls()


# Known state-of-the-art models for each capability
SOTA_MODELS = {
    ModelCapability.TEXT_TO_IMAGE: [
        {
            "name": "FLUX.1-dev",
            "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
            "vram": 12.0,
            "quality": 10,
            "notes": "Best quality, needs 12GB+ VRAM"
        },
        {
            "name": "SDXL",
            "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
            "vram": 8.0,
            "quality": 8,
            "notes": "Great quality, works on 8GB"
        },
        {
            "name": "DreamShaper 8",
            "url": "https://civitai.com/models/4384/dreamshaper",
            "vram": 4.0,
            "quality": 7,
            "notes": "SD1.5, fast and reliable"
        },
    ],
    ModelCapability.TEXT_TO_VIDEO: [
        {
            "name": "LTX Video 13B",
            "url": "https://huggingface.co/Lightricks/LTX-Video",
            "vram": 24.0,
            "quality": 10,
            "notes": "Best quality video, needs 24GB"
        },
        {
            "name": "LTX Video 2B (FP8)",
            "url": "https://huggingface.co/Lightricks/LTX-Video",
            "vram": 6.0,
            "quality": 9,
            "notes": "State-of-the-art for 8GB cards"
        },
        {
            "name": "Hunyuan Video",
            "url": "https://huggingface.co/tencent/HunyuanVideo",
            "vram": 12.0,
            "quality": 9,
            "notes": "Great quality, needs 12GB"
        },
        {
            "name": "AnimateDiff",
            "url": "https://huggingface.co/guoyww/animatediff",
            "vram": 6.0,
            "quality": 7,
            "notes": "Fast, works on 8GB, SD1.5 based"
        },
    ],
    ModelCapability.TEXT_TO_AUDIO: [
        {
            "name": "Stable Audio",
            "url": "https://huggingface.co/stabilityai/stable-audio-open-1.0",
            "vram": 8.0,
            "quality": 9,
            "notes": "High quality audio generation"
        },
        {
            "name": "AudioLDM 2",
            "url": "https://huggingface.co/cvssp/audioldm2",
            "vram": 6.0,
            "quality": 8,
            "notes": "Good audio, works on 8GB"
        },
    ],
}


class ModelRegistry:
    """
    Dynamic registry of available models.
    
    Auto-detects what's installed in ComfyUI and categorizes
    by capability so the MetaPlanner can make informed decisions.
    
    Also knows about online models and hardware capabilities.
    """
    
    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        perplexity_key: Optional[str] = None,
        hardware: Optional[HardwareSpec] = None
    ):
        self.comfyui_url = comfyui_url
        self.perplexity_key = perplexity_key or os.environ.get("PERPLEXITY_API_KEY")
        self.hardware = hardware or HardwareSpec.detect()
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
        lines = [
            "=== Model Registry ===",
            f"Hardware: {self.hardware.gpu_name} ({self.hardware.vram_gb:.1f}GB VRAM)",
            ""
        ]
        
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
    
    def get_missing_capabilities(self) -> List[ModelCapability]:
        """Get capabilities we DON'T have models for."""
        return [
            cap for cap in ModelCapability
            if not self.capabilities.get(cap)
        ]
    
    def get_recommended_models(
        self,
        capability: Optional[ModelCapability] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommended models that would work on this hardware.
        
        Returns models from SOTA_MODELS that:
        1. Fit in available VRAM
        2. Are not already installed
        3. Would improve quality
        """
        recommendations = []
        vram = self.hardware.vram_gb
        
        caps_to_check = [capability] if capability else list(ModelCapability)
        
        for cap in caps_to_check:
            sota = SOTA_MODELS.get(cap, [])
            installed = self.capabilities.get(cap, [])
            installed_quality = max((m.quality_tier for m in installed), default=0)
            
            for model in sota:
                # Skip if won't fit
                if model["vram"] > vram:
                    continue
                
                # Skip if we already have something as good
                if installed_quality >= model["quality"]:
                    continue
                
                # Check if already installed (fuzzy match)
                already_have = any(
                    model["name"].lower().replace(" ", "") in m.name.lower().replace(" ", "")
                    for m in installed
                )
                if already_have:
                    continue
                
                recommendations.append({
                    **model,
                    "capability": cap.value,
                    "would_improve": model["quality"] - installed_quality
                })
        
        # Sort by quality improvement
        recommendations.sort(key=lambda x: x["would_improve"], reverse=True)
        return recommendations
    
    async def research_best_model(
        self, 
        task: str,
        max_vram: Optional[float] = None
    ) -> str:
        """
        Research the best model for a task using Perplexity.
        
        This is how the system learns about NEW state-of-the-art models.
        """
        if not self.perplexity_key:
            return "No Perplexity API key - cannot research models"
        
        vram = max_vram or self.hardware.vram_gb
        
        import urllib.request
        
        query = f"""What is the best AI model for: {task}

Requirements:
- Must work with ComfyUI
- Must fit in {vram}GB VRAM
- Prefer quantized versions (FP8, GGUF) if available

Give me:
1. Model name
2. HuggingFace or CivitAI link
3. VRAM requirement
4. How to install in ComfyUI"""

        data = json.dumps({
            "model": "sonar",
            "messages": [{"role": "user", "content": query}]
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
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Research failed: {e}"
    
    def full_report(self) -> str:
        """Get a complete report of installed vs possible models."""
        lines = [
            "=" * 50,
            "MODEL CAPABILITY REPORT",
            "=" * 50,
            "",
            f"Hardware: {self.hardware.gpu_name}",
            f"VRAM: {self.hardware.vram_gb:.1f} GB",
            "",
            "INSTALLED MODELS:",
            "-" * 30,
        ]
        
        for cap in ModelCapability:
            models = self.capabilities.get(cap, [])
            if models:
                lines.append(f"\n{cap.value}:")
                for m in models:
                    lines.append(f"  [OK] {m.name} (Q{m.quality_tier})")
        
        # Missing capabilities
        missing = self.get_missing_capabilities()
        if missing:
            lines.append("\n" + "-" * 30)
            lines.append("MISSING CAPABILITIES:")
            for cap in missing:
                lines.append(f"  [--] {cap.value}")
        
        # Recommendations
        recs = self.get_recommended_models()
        if recs:
            lines.append("\n" + "-" * 30)
            lines.append("RECOMMENDED UPGRADES:")
            for r in recs[:5]:
                lines.append(f"  >> {r['name']} ({r['capability']})")
                lines.append(f"     VRAM: {r['vram']}GB, Quality: {r['quality']}/10")
                lines.append(f"     {r['url']}")
        
        return "\n".join(lines)
    
    # ==================== COMFY-PILOT INTEGRATION ====================
    
    async def call_comfy_pilot(self, tool: str, **kwargs) -> Dict[str, Any]:
        """Call a Comfy-Pilot tool via the MCP server."""
        async with aiohttp.ClientSession() as session:
            # Comfy-Pilot exposes tools via ComfyUI's API
            # For now, use direct API calls
            try:
                if tool == "download_model":
                    return await self._download_model_direct(**kwargs)
                elif tool == "install_custom_node":
                    return await self._install_node_direct(**kwargs)
                elif tool == "search_custom_nodes":
                    return await self._search_nodes_direct(**kwargs)
                else:
                    return {"error": f"Unknown tool: {tool}"}
            except Exception as e:
                return {"error": str(e)}
    
    async def _download_model_direct(
        self,
        url: str,
        model_type: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> Dict[str, Any]:
        """Download a model using ComfyUI Manager's API."""
        async with aiohttp.ClientSession() as session:
            # ComfyUI Manager endpoint for model download
            payload = {
                "url": url,
                "model_type": model_type
            }
            if filename:
                payload["filename"] = filename
            if subfolder:
                payload["subfolder"] = subfolder
            
            try:
                async with session.post(
                    f"{self.comfyui_url}/manager/model/download",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 min timeout for downloads
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        return {"error": f"HTTP {resp.status}", "body": await resp.text()}
            except Exception as e:
                return {"error": str(e)}
    
    async def _install_node_direct(self, node_id: str) -> Dict[str, Any]:
        """Install a custom node via ComfyUI Manager."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.comfyui_url}/manager/install",
                    json={"id": node_id},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    return await resp.json()
            except Exception as e:
                return {"error": str(e)}
    
    async def _search_nodes_direct(
        self,
        query: Optional[str] = None,
        status: str = "all",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search for custom nodes via ComfyUI Manager."""
        async with aiohttp.ClientSession() as session:
            try:
                params = {"limit": limit}
                if query:
                    params["query"] = query
                if status != "all":
                    params["status"] = status
                
                async with session.get(
                    f"{self.comfyui_url}/manager/search",
                    params=params
                ) as resp:
                    return await resp.json()
            except Exception as e:
                return {"error": str(e)}
    
    async def download_recommended_model(
        self,
        capability: ModelCapability
    ) -> Dict[str, Any]:
        """Download the best recommended model for a capability."""
        recs = self.get_recommended_models(capability)
        if not recs:
            return {"error": f"No recommended models for {capability.value}"}
        
        best = recs[0]
        url = best.get("url", "")
        
        # Map capability to model_type for ComfyUI
        model_type_map = {
            ModelCapability.TEXT_TO_IMAGE: "checkpoints",
            ModelCapability.TEXT_TO_VIDEO: "unet",  # LTX uses UNET
            ModelCapability.TEXT_TO_AUDIO: "checkpoints",
        }
        model_type = model_type_map.get(capability, "checkpoints")
        
        print(f"Downloading {best['name']} from {url}...")
        result = await self._download_model_direct(
            url=url,
            model_type=model_type
        )
        
        if "error" not in result:
            # Refresh registry after download
            await self.refresh()
        
        return result
    
    async def auto_upgrade(self, max_downloads: int = 1) -> List[Dict[str, Any]]:
        """
        Automatically download missing/better models.
        
        This is the AUTONOMOUS capability the paper describes.
        """
        results = []
        recs = self.get_recommended_models()
        
        for rec in recs[:max_downloads]:
            print(f"Auto-upgrading: {rec['name']} for {rec['capability']}")
            result = await self.download_recommended_model(
                ModelCapability(rec['capability'])
            )
            results.append({
                "model": rec['name'],
                "result": result
            })
        
        return results


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
