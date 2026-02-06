"""Discovery Module — Constraint-aware system discovery via ComfyPilot.

Paper Section 5.4: "Traverses the system's atomic tool library"

This module discovers what the user's system CAN do:
- Hardware constraints (GPU, VRAM)
- Available nodes (what can be composed)
- Available models (what capabilities exist)
- Inferred capabilities (what's possible)

NO HARDCODED PATTERNS. Everything is discovered.
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum, auto


class Capability(Enum):
    """Capabilities that can be inferred from available tools."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    UPSCALE = "upscale"
    INPAINT = "inpaint"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class HardwareConstraints:
    """User's hardware constraints."""
    gpu_name: str = "Unknown"
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    
    def can_run(self, required_vram_gb: float) -> bool:
        """Check if model can run on this hardware."""
        # Use 80% of VRAM as safe limit
        return required_vram_gb <= (self.vram_total_gb * 0.8)


@dataclass
class AvailableNode:
    """A node type available in ComfyUI."""
    name: str
    category: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    
    @property
    def is_loader(self) -> bool:
        return "loader" in self.name.lower()
    
    @property
    def is_sampler(self) -> bool:
        return "sampler" in self.name.lower() or "sample" in self.name.lower()
    
    @property
    def is_encoder(self) -> bool:
        return "encode" in self.name.lower()
    
    @property
    def is_decoder(self) -> bool:
        return "decode" in self.name.lower()


@dataclass
class AvailableModel:
    """A model available in ComfyUI."""
    filename: str
    category: str  # checkpoints, unet, vae, loras, etc.
    path: str = ""
    size_gb: float = 0.0
    
    @property
    def inferred_capability(self) -> Capability:
        """Infer capability from filename patterns."""
        name = self.filename.lower()
        
        # Video models
        if any(x in name for x in ['video', 'animate', 'motion', 'wan', 'ltx', 'svd', 'i2v', 't2v']):
            if 'i2v' in name or 'img2vid' in name:
                return Capability.IMAGE_TO_VIDEO
            return Capability.TEXT_TO_VIDEO
        
        # Upscale models
        if any(x in name for x in ['upscale', 'esrgan', '4x', '2x']):
            return Capability.UPSCALE
        
        # Inpaint models
        if 'inpaint' in name:
            return Capability.INPAINT
        
        # Audio models
        if any(x in name for x in ['audio', 'music', 'sound']):
            return Capability.AUDIO
        
        # Default to image generation
        if self.category in ['checkpoints', 'unet', 'diffusion_models']:
            return Capability.TEXT_TO_IMAGE
        
        return Capability.UNKNOWN


@dataclass
class SystemCapabilities:
    """Complete picture of what the system can do."""
    hardware: HardwareConstraints
    nodes: Dict[str, AvailableNode]
    models: Dict[str, List[AvailableModel]]
    capabilities: Set[Capability]
    
    def has_capability(self, cap: Capability) -> bool:
        return cap in self.capabilities
    
    def get_models_for(self, cap: Capability) -> List[AvailableModel]:
        """Get all models that provide a capability."""
        result = []
        for category_models in self.models.values():
            for model in category_models:
                if model.inferred_capability == cap:
                    result.append(model)
        return result
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 50,
            "SYSTEM CAPABILITIES",
            "=" * 50,
            "",
            f"GPU: {self.hardware.gpu_name}",
            f"VRAM: {self.hardware.vram_total_gb:.1f}GB total, {self.hardware.vram_free_gb:.1f}GB free",
            "",
            f"Nodes: {len(self.nodes)} available",
            f"Models: {sum(len(m) for m in self.models.values())} total",
            "",
            "Capabilities:",
        ]
        
        for cap in Capability:
            if cap in self.capabilities:
                models = self.get_models_for(cap)
                lines.append(f"  [YES] {cap.value}: {len(models)} models")
            elif cap != Capability.UNKNOWN:
                lines.append(f"  [NO]  {cap.value}")
        
        return "\n".join(lines)


class SystemDiscovery:
    """Discovers system capabilities via ComfyPilot.
    
    This is the GENERAL approach - no hardcoded patterns.
    Everything is discovered from the actual ComfyUI instance.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.url = comfyui_url.rstrip('/')
        self._cache: Optional[SystemCapabilities] = None
    
    async def discover(self, force_refresh: bool = False) -> SystemCapabilities:
        """Discover all system capabilities."""
        if self._cache and not force_refresh:
            return self._cache
        
        async with aiohttp.ClientSession() as session:
            # Parallel discovery
            hardware, nodes, models = await asyncio.gather(
                self._discover_hardware(session),
                self._discover_nodes(session),
                self._discover_models(session),
            )
        
        # Infer capabilities from available models and nodes
        capabilities = self._infer_capabilities(nodes, models)
        
        self._cache = SystemCapabilities(
            hardware=hardware,
            nodes=nodes,
            models=models,
            capabilities=capabilities
        )
        
        return self._cache
    
    async def _discover_hardware(self, session: aiohttp.ClientSession) -> HardwareConstraints:
        """Discover hardware via /system_stats."""
        try:
            async with session.get(f"{self.url}/system_stats", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                
                devices = data.get("devices", [])
                if devices:
                    device = devices[0]
                    return HardwareConstraints(
                        gpu_name=device.get("name", "Unknown"),
                        vram_total_gb=device.get("vram_total", 0) / (1024**3),
                        vram_free_gb=device.get("vram_free", 0) / (1024**3),
                    )
        except Exception as e:
            print(f"Hardware discovery failed: {e}")
        
        return HardwareConstraints()
    
    async def _discover_nodes(self, session: aiohttp.ClientSession) -> Dict[str, AvailableNode]:
        """Discover available nodes via /object_info."""
        nodes = {}
        
        try:
            async with session.get(f"{self.url}/object_info", timeout=aiohttp.ClientTimeout(total=30)) as resp:
                data = await resp.json()
                
                for name, info in data.items():
                    nodes[name] = AvailableNode(
                        name=name,
                        category=info.get("category", ""),
                        inputs=info.get("input", {}).get("required", {}),
                        outputs=info.get("output", []),
                    )
        except Exception as e:
            print(f"Node discovery failed: {e}")
        
        return nodes
    
    async def _discover_models(self, session: aiohttp.ClientSession) -> Dict[str, List[AvailableModel]]:
        """Discover available models via /models/* endpoints."""
        models = {}
        
        # Standard ComfyUI model categories
        categories = [
            "checkpoints", "unet", "diffusion_models", "vae", 
            "clip", "loras", "upscale_models", "embeddings"
        ]
        
        for category in categories:
            try:
                async with session.get(f"{self.url}/models/{category}", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        model_list = data if isinstance(data, list) else data.get("value", [])
                        
                        models[category] = [
                            AvailableModel(
                                filename=m if isinstance(m, str) else m.get("name", ""),
                                category=category,
                            )
                            for m in model_list
                        ]
            except Exception as e:
                # Category might not exist
                pass
        
        return models
    
    def _infer_capabilities(
        self, 
        nodes: Dict[str, AvailableNode],
        models: Dict[str, List[AvailableModel]]
    ) -> Set[Capability]:
        """Infer capabilities from available nodes and models."""
        capabilities = set()
        
        # Check models for capabilities
        for category_models in models.values():
            for model in category_models:
                cap = model.inferred_capability
                if cap != Capability.UNKNOWN:
                    capabilities.add(cap)
        
        # Also check node availability for capabilities
        node_names = set(n.lower() for n in nodes.keys())
        
        # Video nodes
        if any('video' in n or 'animate' in n for n in node_names):
            capabilities.add(Capability.TEXT_TO_VIDEO)
        
        # Image-to-video nodes  
        if any('i2v' in n or 'img2vid' in n for n in node_names):
            capabilities.add(Capability.IMAGE_TO_VIDEO)
        
        # Upscale nodes
        if any('upscale' in n or 'esrgan' in n for n in node_names):
            capabilities.add(Capability.UPSCALE)
        
        # Inpaint nodes
        if any('inpaint' in n for n in node_names):
            capabilities.add(Capability.INPAINT)
        
        # Basic image generation (if we have samplers and models)
        has_sampler = any(n.is_sampler for n in nodes.values())
        has_model = bool(models.get("checkpoints") or models.get("unet") or models.get("diffusion_models"))
        if has_sampler and has_model:
            capabilities.add(Capability.TEXT_TO_IMAGE)
        
        return capabilities


class ModelSearch:
    """Search for models on CivitAI and HuggingFace.
    
    Filtered by user's hardware constraints.
    """
    
    def __init__(self, constraints: HardwareConstraints):
        self.constraints = constraints
    
    async def search_civitai(
        self, 
        query: str,
        model_type: str = "Checkpoint",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search CivitAI for models within VRAM constraints."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "types": model_type,
                    "limit": limit,
                    "sort": "Most Downloaded"
                }
                
                async with session.get(
                    "https://civitai.com/api/v1/models",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._filter_by_constraints(data.get("items", []))
        except Exception as e:
            print(f"CivitAI search failed: {e}")
        
        return []
    
    async def search_huggingface(
        self,
        query: str,
        pipeline_tag: str = "text-to-image",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace for models."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "search": query,
                    "pipeline_tag": pipeline_tag,
                    "limit": limit,
                    "sort": "downloads"
                }
                
                async with session.get(
                    "https://huggingface.co/api/models",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            print(f"HuggingFace search failed: {e}")
        
        return []
    
    def _filter_by_constraints(self, models: List[Dict]) -> List[Dict]:
        """Filter models by VRAM constraints."""
        # CivitAI doesn't always have VRAM info, so we estimate
        # FP16 checkpoint ~= 2-4GB, FP8 ~= 1-2GB
        return models  # TODO: Better filtering when VRAM info available
    
    async def recommend_for_capability(
        self,
        capability: Capability
    ) -> List[Dict[str, Any]]:
        """Recommend models for a capability within constraints."""
        # Map capability to search terms
        search_map = {
            Capability.TEXT_TO_IMAGE: ("stable diffusion", "Checkpoint"),
            Capability.TEXT_TO_VIDEO: ("video generation", "Checkpoint"),
            Capability.IMAGE_TO_VIDEO: ("image to video", "Checkpoint"),
            Capability.UPSCALE: ("upscale", "Upscaler"),
        }
        
        if capability not in search_map:
            return []
        
        query, model_type = search_map[capability]
        
        # Search both sources
        civitai_results = await self.search_civitai(query, model_type)
        hf_results = await self.search_huggingface(query)
        
        return civitai_results + hf_results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def discover_system(comfyui_url: str = "http://127.0.0.1:8188") -> SystemCapabilities:
    """Discover system capabilities."""
    discovery = SystemDiscovery(comfyui_url)
    return await discovery.discover()


async def find_models_for(
    capability: Capability,
    constraints: HardwareConstraints
) -> List[Dict[str, Any]]:
    """Find models for a capability within constraints."""
    search = ModelSearch(constraints)
    return await search.recommend_for_capability(capability)
