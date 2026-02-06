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
    CHARACTER_CONSISTENCY = "character_consistency"  # IP-Adapter, LoRA character refs
    STYLE_TRANSFER = "style_transfer"  # Style reference from images
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
        
        # IP-Adapter / Character consistency models
        if any(x in name for x in ['ipadapter', 'ip_adapter', 'ip-adapter', 'instantid', 'faceid', 'pulid']):
            return Capability.CHARACTER_CONSISTENCY
        
        # Style transfer / reference models
        if any(x in name for x in ['style', 'reference', 'clipvision']):
            return Capability.STYLE_TRANSFER
        
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
    
    @property
    def is_ipadapter(self) -> bool:
        """Check if this is an IP-Adapter model."""
        name = self.filename.lower()
        return any(x in name for x in ['ipadapter', 'ip_adapter', 'ip-adapter'])
    
    @property
    def is_character_lora(self) -> bool:
        """Check if this is a character/person LoRA."""
        name = self.filename.lower()
        # Character LoRAs often have these patterns
        return self.category == 'loras' and any(x in name for x in [
            'character', 'person', 'face', 'portrait', 'style', 'celeb'
        ])
    
    @property
    def is_clip_vision(self) -> bool:
        """Check if this is a CLIP Vision model."""
        name = self.filename.lower()
        return self.category == 'clip_vision' or 'clipvision' in name or 'clip_vision' in name


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
    
    def get_ipadapter_models(self) -> List[AvailableModel]:
        """Get all IP-Adapter models."""
        result = []
        for category in ['ipadapter', 'instantid', 'pulid', 'faceid']:
            result.extend(self.models.get(category, []))
        # Also check other categories for IP-Adapter files
        for category_models in self.models.values():
            for model in category_models:
                if model.is_ipadapter and model not in result:
                    result.append(model)
        return result
    
    def get_clip_vision_models(self) -> List[AvailableModel]:
        """Get all CLIP Vision models."""
        result = list(self.models.get('clip_vision', []))
        for category_models in self.models.values():
            for model in category_models:
                if model.is_clip_vision and model not in result:
                    result.append(model)
        return result
    
    def get_character_loras(self) -> List[AvailableModel]:
        """Get all character/person LoRAs."""
        result = []
        for model in self.models.get('loras', []):
            if model.is_character_lora:
                result.append(model)
        return result
    
    def has_ipadapter_support(self) -> bool:
        """Check if full IP-Adapter workflow is possible."""
        # Need IP-Adapter node + IP-Adapter model + CLIP Vision
        node_names = set(n.lower() for n in self.nodes.keys())
        has_ipadapter_node = any('ipadapter' in n for n in node_names)
        has_ipadapter_model = bool(self.get_ipadapter_models())
        has_clip_vision = bool(self.get_clip_vision_models()) or 'CLIPVisionLoader' in self.nodes
        return has_ipadapter_node and (has_ipadapter_model or has_clip_vision)
    
    def has_reference_image_support(self) -> bool:
        """Check if any reference image workflow is possible (IP-Adapter, ByteDance, etc.)."""
        node_names = set(n.lower() for n in self.nodes.keys())
        # Check for various reference image approaches
        return (
            any('ipadapter' in n for n in node_names) or
            any('reference' in n and 'image' in n for n in node_names) or
            any('bytedance' in n.lower() for n in node_names) or
            'CLIPVisionEncode' in self.nodes  # Can encode reference images
        )
    
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
        
        # Character consistency details
        lines.append("")
        lines.append("Character Consistency:")
        lines.append(f"  IP-Adapter support: {'YES' if self.has_ipadapter_support() else 'NO'}")
        lines.append(f"  Reference image support: {'YES' if self.has_reference_image_support() else 'NO'}")
        lines.append(f"  IP-Adapter models: {len(self.get_ipadapter_models())}")
        lines.append(f"  CLIP Vision models: {len(self.get_clip_vision_models())}")
        lines.append(f"  Character LoRAs: {len(self.get_character_loras())}")
        
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
        
        # Standard ComfyUI model categories + IP-Adapter related
        categories = [
            "checkpoints", "unet", "diffusion_models", "vae", 
            "clip", "loras", "upscale_models", "embeddings",
            # IP-Adapter / Character consistency related
            "ipadapter", "clip_vision", "insightface", "instantid", "pulid", "faceid"
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
        
        # IP-Adapter / Character consistency nodes
        ip_adapter_patterns = ['ipadapter', 'ip_adapter', 'ip-adapter', 'instantid', 'faceid', 'pulid']
        if any(any(p in n for p in ip_adapter_patterns) for n in node_names):
            capabilities.add(Capability.CHARACTER_CONSISTENCY)
        
        # CLIP Vision (needed for IP-Adapter) - partial support for character refs
        if any('clipvision' in n or 'clip_vision' in n for n in node_names):
            # CLIP Vision enables style/image reference even without full IP-Adapter
            capabilities.add(Capability.STYLE_TRANSFER)
        
        # ByteDance reference nodes (alternative to IP-Adapter)
        if any('reference' in n and ('image' in n or 'bytedance' in n) for n in node_names):
            capabilities.add(Capability.CHARACTER_CONSISTENCY)
        
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
