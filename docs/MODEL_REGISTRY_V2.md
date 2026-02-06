# ModelRegistry V2 — Discovery-Based Architecture

## Design Principles

1. **Query, don't assume** — Ask ComfyUI what it can do
2. **Nodes define capabilities** — If a node can load it, the node knows what it does
3. **Remote-first** — Always query the target system, not localhost
4. **Graceful degradation** — Unknown models are tracked, not ignored
5. **Static patterns are fallback** — Last resort, not first choice

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      ModelRegistry V2                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Hardware   │    │    Model     │    │  Capability  │     │
│  │   Detector   │    │   Scanner    │    │   Inferrer   │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                   │                   │              │
│         v                   v                   v              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    ComfyUI Client                        │  │
│  │  • /system_stats  → GPU, VRAM, OS                       │  │
│  │  • /object_info   → All nodes + their input specs       │  │
│  │  • /models/*      → All model files by category         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Hardware Detector

```python
@dataclass
class HardwareSpec:
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    os: str
    comfyui_version: str
    is_remote: bool

class HardwareDetector:
    """Detects hardware from the TARGET system, not local."""
    
    def __init__(self, comfyui_url: str):
        self.url = comfyui_url
        self.is_remote = not self._is_localhost(comfyui_url)
    
    async def detect(self) -> HardwareSpec:
        """Query the ComfyUI instance for its hardware."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/system_stats") as resp:
                stats = await resp.json()
        
        device = stats["devices"][0]  # Primary GPU
        system = stats["system"]
        
        return HardwareSpec(
            gpu_name=device["name"].split(":")[-1].strip(),  # Parse "cuda:0 NVIDIA GeForce RTX 3090"
            vram_total_gb=device["vram_total"] / (1024**3),
            vram_free_gb=device["vram_free"] / (1024**3),
            ram_total_gb=system["ram_total"] / (1024**3),
            os=system["os"],
            comfyui_version=system["comfyui_version"],
            is_remote=self.is_remote
        )
    
    def _is_localhost(self, url: str) -> bool:
        return any(x in url for x in ["localhost", "127.0.0.1", "0.0.0.0"])
```

---

## Component 2: Model Scanner

```python
class ModelScanner:
    """Scans all model directories from ComfyUI."""
    
    # ComfyUI model categories
    CATEGORIES = [
        "checkpoints",
        "unet", 
        "diffusion_models",
        "vae",
        "loras",
        "controlnet",
        "clip",
        "clip_vision",
        "upscale_models",
        "embeddings",
    ]
    
    async def scan_all(self) -> Dict[str, List[str]]:
        """Get all models from all categories."""
        models = {}
        
        async with aiohttp.ClientSession() as session:
            for category in self.CATEGORIES:
                try:
                    async with session.get(f"{self.url}/models/{category}") as resp:
                        if resp.status == 200:
                            models[category] = await resp.json()
                except:
                    pass  # Category might not exist
        
        return models
    
    async def get_object_info(self) -> Dict[str, Any]:
        """Get full node info — this tells us what nodes load what models."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/object_info") as resp:
                return await resp.json()
```

---

## Component 3: Capability Inferrer (THE KEY INNOVATION)

```python
class CapabilityInferrer:
    """
    Infers model capabilities by analyzing which nodes can load them.
    
    This is the key insight: ComfyUI nodes KNOW what models they work with.
    If WanVideoSampler lists a model, it's a Wan video model.
    """
    
    # Node → Capability mapping
    # These are the LOADER nodes that tell us what a model does
    NODE_CAPABILITY_MAP = {
        # Video generation
        "LTXVLoader": [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO],
        "WanVideoLoader": [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO],
        "CogVideoLoader": [Capability.TEXT_TO_VIDEO],
        "ADE_AnimateDiffLoaderWithContext": [Capability.TEXT_TO_VIDEO],
        "AnimateDiffLoaderV1": [Capability.TEXT_TO_VIDEO],
        
        # Image generation
        "CheckpointLoaderSimple": [Capability.TEXT_TO_IMAGE],
        "UNETLoader": [Capability.TEXT_TO_IMAGE, Capability.TEXT_TO_VIDEO],  # Context-dependent
        
        # Image-to-image
        "ControlNetLoader": [Capability.IMAGE_TO_IMAGE],
        "IPAdapterModelLoader": [Capability.IMAGE_TO_IMAGE],
        
        # Upscaling
        "UpscaleModelLoader": [Capability.UPSCALE],
        
        # Audio (if supported)
        "AudioModelLoader": [Capability.TEXT_TO_AUDIO],
    }
    
    # Quality indicators in node names or model names
    QUALITY_HINTS = {
        "high": 9, "hq": 9, "best": 10,
        "low": 5, "fast": 5, "turbo": 6, "lcm": 6,
        "fp8": 7, "fp16": 8, "bf16": 8, "fp32": 9,
        "distilled": 7, "pruned": 6,
    }
    
    def __init__(self, object_info: Dict[str, Any]):
        self.object_info = object_info
        self._build_model_to_node_map()
    
    def _build_model_to_node_map(self):
        """
        Build reverse index: model_filename → [nodes that can load it]
        
        This is the magic. We look at every node's input spec,
        find which models it accepts, and build the mapping.
        """
        self.model_to_nodes: Dict[str, List[str]] = defaultdict(list)
        
        for node_name, node_info in self.object_info.items():
            inputs = node_info.get("input", {}).get("required", {})
            
            # Look for model selection inputs
            for input_name, input_spec in inputs.items():
                if self._is_model_input(input_name):
                    # input_spec[0] is the list of valid values
                    if isinstance(input_spec, list) and len(input_spec) > 0:
                        models = input_spec[0] if isinstance(input_spec[0], list) else []
                        for model in models:
                            self.model_to_nodes[model].append(node_name)
    
    def _is_model_input(self, name: str) -> bool:
        """Check if an input is a model selector."""
        model_inputs = [
            "ckpt_name", "unet_name", "vae_name", "lora_name",
            "model_name", "control_net_name", "clip_name",
            "upscale_model", "ipadapter_file"
        ]
        return name in model_inputs
    
    def infer_capabilities(self, model_filename: str) -> List[Capability]:
        """
        Infer what a model can do based on which nodes accept it.
        """
        nodes = self.model_to_nodes.get(model_filename, [])
        
        capabilities = set()
        for node in nodes:
            if node in self.NODE_CAPABILITY_MAP:
                capabilities.update(self.NODE_CAPABILITY_MAP[node])
        
        # If no specific nodes found, try pattern matching as fallback
        if not capabilities:
            capabilities = self._fallback_pattern_match(model_filename)
        
        return list(capabilities) if capabilities else [Capability.UNKNOWN]
    
    def infer_quality(self, model_filename: str) -> int:
        """Infer quality tier from filename hints."""
        lower = model_filename.lower()
        
        for hint, quality in self.QUALITY_HINTS.items():
            if hint in lower:
                return quality
        
        return 7  # Default middle-tier
    
    def infer_vram(self, model_filename: str, category: str) -> float:
        """Estimate VRAM from filename and category."""
        lower = model_filename.lower()
        
        # Size indicators
        if "14b" in lower:
            base = 14.0
        elif "13b" in lower:
            base = 12.0
        elif "7b" in lower:
            base = 7.0
        elif "2b" in lower:
            base = 3.0
        elif "1b" in lower:
            base = 2.0
        else:
            base = 4.0  # Default
        
        # Quantization reduces VRAM
        if "fp8" in lower or "q8" in lower:
            base *= 0.5
        elif "q4" in lower:
            base *= 0.3
        elif "fp16" in lower:
            base *= 0.8
        
        return round(base, 1)
    
    def _fallback_pattern_match(self, filename: str) -> Set[Capability]:
        """Last resort: pattern matching."""
        lower = filename.lower()
        
        PATTERNS = {
            ("ltx", "ltxv"): {Capability.TEXT_TO_VIDEO},
            ("wan2", "wanvideo"): {Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO},
            ("anisora",): {Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO},
            ("animatediff", "mm_sd"): {Capability.TEXT_TO_VIDEO},
            ("cogvideo",): {Capability.TEXT_TO_VIDEO},
            ("flux",): {Capability.TEXT_TO_IMAGE},
            ("sdxl", "sd_xl"): {Capability.TEXT_TO_IMAGE},
            ("hidream",): {Capability.TEXT_TO_IMAGE},
            ("upscale", "esrgan", "realesrgan"): {Capability.UPSCALE},
            ("controlnet", "canny", "depth"): {Capability.IMAGE_TO_IMAGE},
            ("ipadapter",): {Capability.IMAGE_TO_IMAGE},
        }
        
        caps = set()
        for patterns, capabilities in PATTERNS.items():
            if any(p in lower for p in patterns):
                caps.update(capabilities)
        
        return caps
```

---

## Component 4: Unified Registry

```python
@dataclass
class ModelSpec:
    """Full specification for a discovered model."""
    filename: str
    name: str  # Human-readable
    category: str  # checkpoints, unet, etc.
    capabilities: List[Capability]
    vram_required: float
    quality_tier: int  # 1-10
    loader_nodes: List[str]  # Which nodes can load this
    notes: str = ""

class ModelRegistryV2:
    """
    Discovery-based model registry.
    
    Queries ComfyUI to understand:
    1. What hardware is available
    2. What models are installed  
    3. What each model can do (via node analysis)
    """
    
    def __init__(self, comfyui_url: str):
        self.url = comfyui_url
        self.hardware: Optional[HardwareSpec] = None
        self.models: Dict[str, ModelSpec] = {}
        self.capabilities: Dict[Capability, List[ModelSpec]] = defaultdict(list)
    
    async def discover(self) -> None:
        """
        Full discovery process. Call this once at startup.
        """
        # 1. Detect hardware (from TARGET system)
        detector = HardwareDetector(self.url)
        self.hardware = await detector.detect()
        print(f"🖥️  Hardware: {self.hardware.gpu_name} ({self.hardware.vram_total_gb:.1f}GB)")
        
        # 2. Scan all models
        scanner = ModelScanner(self.url)
        all_models = await scanner.scan_all()
        object_info = await scanner.get_object_info()
        
        # 3. Infer capabilities
        inferrer = CapabilityInferrer(object_info)
        
        # 4. Build registry
        for category, filenames in all_models.items():
            for filename in filenames:
                caps = inferrer.infer_capabilities(filename)
                quality = inferrer.infer_quality(filename)
                vram = inferrer.infer_vram(filename, category)
                nodes = inferrer.model_to_nodes.get(filename, [])
                
                spec = ModelSpec(
                    filename=filename,
                    name=self._clean_name(filename),
                    category=category,
                    capabilities=caps,
                    vram_required=vram,
                    quality_tier=quality,
                    loader_nodes=nodes,
                )
                
                self.models[filename] = spec
                
                for cap in caps:
                    self.capabilities[cap].append(spec)
        
        print(f"📦 Discovered {len(self.models)} models")
        for cap, models in self.capabilities.items():
            print(f"   {cap.value}: {len(models)} models")
    
    def get_best_for(
        self, 
        capability: Capability,
        max_vram: Optional[float] = None,
        prefer: str = "quality"  # or "speed" or "balanced"
    ) -> Optional[ModelSpec]:
        """
        Get the best model for a capability, respecting constraints.
        """
        candidates = self.capabilities.get(capability, [])
        
        if not candidates:
            return None
        
        # Filter by VRAM
        if max_vram is None:
            max_vram = self.hardware.vram_free_gb if self.hardware else 8.0
        
        candidates = [m for m in candidates if m.vram_required <= max_vram]
        
        if not candidates:
            return None
        
        # Sort by preference
        if prefer == "quality":
            candidates.sort(key=lambda m: m.quality_tier, reverse=True)
        elif prefer == "speed":
            candidates.sort(key=lambda m: m.vram_required)  # Lower VRAM = faster
        else:  # balanced
            candidates.sort(key=lambda m: m.quality_tier / m.vram_required, reverse=True)
        
        return candidates[0]
    
    def _clean_name(self, filename: str) -> str:
        """Convert filename to human-readable name."""
        name = filename.replace(".safetensors", "").replace(".ckpt", "")
        name = name.replace("_", " ").replace("-", " ")
        # Remove common suffixes
        for suffix in ["fp8", "fp16", "bf16", "e4m3fn", "e5m2", "scaled", "KJ"]:
            name = name.replace(suffix, "")
        return name.strip().title()
    
    def status(self) -> str:
        """Pretty print registry status."""
        lines = [
            "=" * 50,
            "MODEL REGISTRY STATUS",
            "=" * 50,
            "",
            f"Target: {self.url}",
            f"Hardware: {self.hardware.gpu_name}" if self.hardware else "Hardware: Unknown",
            f"VRAM: {self.hardware.vram_total_gb:.1f}GB total, {self.hardware.vram_free_gb:.1f}GB free" if self.hardware else "",
            "",
            f"Total Models: {len(self.models)}",
            "",
        ]
        
        for cap in Capability:
            models = self.capabilities.get(cap, [])
            if models:
                lines.append(f"{cap.value} ({len(models)} models):")
                for m in sorted(models, key=lambda x: x.quality_tier, reverse=True)[:3]:
                    lines.append(f"  • {m.name} (Q{m.quality_tier}, {m.vram_required}GB)")
                if len(models) > 3:
                    lines.append(f"  ... and {len(models) - 3} more")
                lines.append("")
        
        return "\n".join(lines)
```

---

## Usage

```python
async def main():
    # Connect to remote ComfyUI
    registry = ModelRegistryV2("http://192.168.1.143:8188")
    
    # Discover everything
    await registry.discover()
    
    # Print status
    print(registry.status())
    
    # Get best video model that fits in VRAM
    best_video = registry.get_best_for(
        Capability.TEXT_TO_VIDEO,
        max_vram=registry.hardware.vram_free_gb,
        prefer="quality"
    )
    
    if best_video:
        print(f"Best video model: {best_video.name}")
        print(f"  File: {best_video.filename}")
        print(f"  VRAM: {best_video.vram_required}GB")
        print(f"  Quality: {best_video.quality_tier}/10")
        print(f"  Loader nodes: {best_video.loader_nodes}")
```

---

## Key Differences from V1

| Aspect | V1 (Current) | V2 (Proposed) |
|--------|-------------|---------------|
| Model recognition | Pattern matching on filenames | Node compatibility analysis |
| Hardware detection | Local nvidia-smi | Remote /system_stats |
| Unknown models | Ignored | Tracked as UNKNOWN |
| Capability source | Hardcoded KNOWN_MODELS | Inferred from node graph |
| Maintenance | Manual pattern updates | Self-discovering |
| Extensibility | Add patterns | Automatic with new nodes |

---

## Migration Path

1. **Phase 1**: Add remote hardware detection (non-breaking)
2. **Phase 2**: Add CapabilityInferrer alongside existing patterns
3. **Phase 3**: Make inference primary, patterns fallback
4. **Phase 4**: Remove hardcoded patterns entirely

This way we can ship incremental improvements without breaking existing functionality.
