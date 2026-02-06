"""Workflow Executor — Discovery-based execution engine.

Implements the paper's architecture:
1. DISCOVER what's available (hardware + models)
2. SELECT the best strategy for the task
3. BUILD the workflow dynamically  
4. EXECUTE with observation and feedback

This replaces hardcoded workflows with adaptive, discoverable generation.
"""

import asyncio
import aiohttp
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

from .workflow_strategies import (
    Capability,
    ModelSpec,
    VideoRequest,
    WorkflowResult,
    VideoGenerationStrategy,
    StrategyFactory,
    ExecutionObserver,
    LoggingObserver,
)


# =============================================================================
# DISCOVERY — Query ComfyUI for capabilities
# =============================================================================

@dataclass
class HardwareSpec:
    """Hardware specification from target system."""
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    os: str
    comfyui_version: str


class ComfyUIDiscovery:
    """Discovers capabilities from a ComfyUI instance.
    
    Queries:
    - /system_stats for hardware info
    - /object_info for available nodes
    - /models/* for available models
    """
    
    MODEL_CATEGORIES = [
        "checkpoints", "unet", "diffusion_models", "vae",
        "clip", "text_encoders", "loras", "controlnet",
        "upscale_models", "embeddings"
    ]
    
    # Node -> Capability mapping for inference
    NODE_CAPABILITIES = {
        # Video
        "EmptyLTXVLatentVideo": [Capability.TEXT_TO_VIDEO],
        "LTXVConditioning": [Capability.TEXT_TO_VIDEO],
        "WanVideoSampler": [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO],
        "WanImageToVideo": [Capability.IMAGE_TO_VIDEO],
        "ADE_AnimateDiffLoaderWithContext": [Capability.TEXT_TO_VIDEO],
        "AnimateDiffLoaderV1": [Capability.TEXT_TO_VIDEO],
        "CogVideoSampler": [Capability.TEXT_TO_VIDEO],
        
        # Image
        "CheckpointLoaderSimple": [Capability.TEXT_TO_IMAGE],
        "KSampler": [Capability.TEXT_TO_IMAGE],
        
        # Upscale
        "UpscaleModelLoader": [Capability.UPSCALE],
        "ImageUpscaleWithModel": [Capability.UPSCALE],
    }
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.hardware: Optional[HardwareSpec] = None
        self.nodes: Dict[str, Any] = {}
        self.models: Dict[str, List[ModelSpec]] = defaultdict(list)
        self.capabilities: Dict[Capability, List[ModelSpec]] = defaultdict(list)
    
    async def discover(self) -> None:
        """Run full discovery process."""
        async with aiohttp.ClientSession() as session:
            # 1. Hardware
            self.hardware = await self._discover_hardware(session)
            print(f"Hardware: {self.hardware.gpu_name} ({self.hardware.vram_total_gb:.1f}GB)")
            
            # 2. Nodes (tells us what the system can do)
            self.nodes = await self._discover_nodes(session)
            print(f"Nodes: {len(self.nodes)} available")
            
            # 3. Models
            await self._discover_models(session)
            print(f"Models: {sum(len(v) for v in self.models.values())} total")
            
            # 4. Infer capabilities
            self._infer_capabilities()
            for cap, models in self.capabilities.items():
                if models:
                    print(f"  {cap.value}: {len(models)} models")
    
    async def _discover_hardware(self, session: aiohttp.ClientSession) -> HardwareSpec:
        """Query hardware from /system_stats."""
        try:
            async with session.get(
                f"{self.base_url}/system_stats",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
            
            device = data["devices"][0]
            system = data["system"]
            
            return HardwareSpec(
                gpu_name=device["name"].split(":")[-1].strip() if ":" in device["name"] else device["name"],
                vram_total_gb=device["vram_total"] / (1024**3),
                vram_free_gb=device["vram_free"] / (1024**3),
                ram_total_gb=system["ram_total"] / (1024**3),
                os=system["os"],
                comfyui_version=system.get("comfyui_version", "unknown"),
            )
        except Exception as e:
            print(f"Warning: Could not get hardware info: {e}")
            return HardwareSpec(
                gpu_name="Unknown",
                vram_total_gb=8.0,
                vram_free_gb=8.0,
                ram_total_gb=16.0,
                os="unknown",
                comfyui_version="unknown",
            )
    
    async def _discover_nodes(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Query available nodes from /object_info."""
        try:
            async with session.get(
                f"{self.base_url}/object_info",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                return await resp.json()
        except Exception as e:
            print(f"Warning: Could not get node info: {e}")
            return {}
    
    async def _discover_models(self, session: aiohttp.ClientSession) -> None:
        """Query available models from /models/*."""
        for category in self.MODEL_CATEGORIES:
            try:
                async with session.get(
                    f"{self.base_url}/models/{category}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        filenames = await resp.json()
                        for filename in filenames:
                            spec = self._create_model_spec(filename, category)
                            self.models[category].append(spec)
            except Exception:
                pass  # Category might not exist
    
    def _create_model_spec(self, filename: str, category: str) -> ModelSpec:
        """Create ModelSpec from filename with inferred properties."""
        lower = filename.lower()
        
        # Infer capabilities from filename patterns
        capabilities = []
        if any(x in lower for x in ["ltx", "ltxv"]):
            capabilities.append(Capability.TEXT_TO_VIDEO)
        if any(x in lower for x in ["wan", "anisora"]):
            capabilities.extend([Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO])
        if any(x in lower for x in ["animatediff", "mm_sd"]):
            capabilities.append(Capability.TEXT_TO_VIDEO)
        if any(x in lower for x in ["flux", "sdxl", "sd_xl", "stable"]):
            capabilities.append(Capability.TEXT_TO_IMAGE)
        if any(x in lower for x in ["upscale", "esrgan"]):
            capabilities.append(Capability.UPSCALE)
        
        if not capabilities:
            capabilities = [Capability.UNKNOWN]
        
        # Infer VRAM requirement
        vram = 4.0  # Default
        if "19b" in lower or "14b" in lower:
            vram = 12.0
        elif "7b" in lower:
            vram = 6.0
        elif "2b" in lower or "1b" in lower:
            vram = 3.0
        
        if "fp8" in lower or "q8" in lower:
            vram *= 0.5
        elif "q4" in lower:
            vram *= 0.3
        
        # Infer quality tier
        quality = 7
        
        # Model size matters: larger = better quality potential
        if "19b" in lower:
            quality = 9  # State-of-the-art size
        elif "14b" in lower or "13b" in lower:
            quality = 8
        elif "7b" in lower:
            quality = 7
        elif "2b" in lower or "1b" in lower:
            quality = 6
        
        # Quality hints in filename
        if "high" in lower or "hq" in lower:
            quality = max(quality, 8)  # Don't downgrade from size-based
        elif "low" in lower or "fast" in lower or "turbo" in lower:
            quality = min(quality, 6)
        
        # State-of-the-art models get bonus
        if "ltx-2" in lower or "ltx2" in lower:
            quality = max(quality, 9)  # LTX-2 is current SOTA for video
        
        # Find loader nodes
        loaders = self._find_loader_nodes(filename, category)
        
        return ModelSpec(
            filename=filename,
            category=category,
            capabilities=capabilities,
            vram_required=round(vram, 1),
            quality_tier=quality,
            loader_nodes=loaders,
        )
    
    def _find_loader_nodes(self, filename: str, category: str) -> List[str]:
        """Find which nodes can load this model."""
        loaders = []
        
        # Map category to expected input names
        input_names = {
            "checkpoints": ["ckpt_name"],
            "unet": ["unet_name"],
            "diffusion_models": ["unet_name", "model_name"],
            "vae": ["vae_name"],
            "clip": ["clip_name"],
            "text_encoders": ["clip_name", "text_encoder"],
            "loras": ["lora_name"],
            "controlnet": ["control_net_name"],
            "upscale_models": ["upscale_model", "model_name"],
        }
        
        expected_inputs = input_names.get(category, [])
        
        for node_name, node_info in self.nodes.items():
            inputs = node_info.get("input", {}).get("required", {})
            for input_name in expected_inputs:
                if input_name in inputs:
                    # Check if this model is in the allowed values
                    spec = inputs[input_name]
                    if isinstance(spec, list) and len(spec) > 0:
                        allowed = spec[0] if isinstance(spec[0], list) else []
                        if filename in allowed:
                            loaders.append(node_name)
                            break
        
        return loaders
    
    def _infer_capabilities(self) -> None:
        """Build capability -> models index."""
        for category, models in self.models.items():
            for model in models:
                for cap in model.capabilities:
                    self.capabilities[cap].append(model)
    
    def has_capability(self, capability: Capability) -> bool:
        """Check if any model provides this capability."""
        return len(self.capabilities.get(capability, [])) > 0
    
    def get_models_for(self, capability: Capability) -> List[ModelSpec]:
        """Get all models that provide a capability."""
        return self.capabilities.get(capability, [])


# =============================================================================
# EXECUTOR — Strategy-based workflow execution
# =============================================================================

class WorkflowExecutor:
    """Executes workflows using discovered strategies.
    
    Workflow:
    1. discover() - Find what's available
    2. select_strategy() - Pick best approach  
    3. execute() - Run with observation
    """
    
    def __init__(self, comfyui_url: str):
        self.url = comfyui_url.rstrip("/")
        self.discovery = ComfyUIDiscovery(comfyui_url)
        self.strategy: Optional[VideoGenerationStrategy] = None
        self.selected_models: Dict[str, ModelSpec] = {}
        self.observers: List[ExecutionObserver] = [LoggingObserver()]
        self._discovered = False
    
    async def discover(self) -> None:
        """Run discovery to find available models and capabilities."""
        await self.discovery.discover()
        self._discovered = True
    
    def select_strategy(
        self,
        capability: Capability = Capability.TEXT_TO_VIDEO,
        preference: str = "quality",
        max_vram: Optional[float] = None
    ) -> Optional[VideoGenerationStrategy]:
        """Select the best strategy for a capability.
        
        Args:
            capability: What you want to do
            preference: "quality", "speed", or "balanced"
            max_vram: Override VRAM limit. Default uses TOTAL VRAM (hardware capability),
                      not free VRAM (transient state). Models can be unloaded.
        """
        if not self._discovered:
            raise RuntimeError("Must call discover() first")
        
        # Use TOTAL VRAM as constraint, not free VRAM
        # Rationale: Free VRAM is transient state. ComfyUI can unload models.
        # We want to select based on what the hardware CAN do, not current state.
        if max_vram is None:
            max_vram = self.discovery.hardware.vram_total_gb if self.discovery.hardware else 8.0
        
        self.strategy = StrategyFactory.create(
            available_models=self.discovery.models,
            max_vram=max_vram,
            preference=preference,
        )
        
        if self.strategy:
            self.selected_models = self.strategy.select_models(
                self.discovery.models, max_vram
            )
            print(f"Selected strategy: {self.strategy.name}")
            for role, model in self.selected_models.items():
                print(f"  {role}: {model.filename}")
        else:
            print(f"No strategy available for {capability.value}")
        
        return self.strategy
    
    async def execute(self, request: VideoRequest) -> WorkflowResult:
        """Execute video generation with selected strategy."""
        if not self.strategy:
            return WorkflowResult(
                success=False,
                error="No strategy selected. Call select_strategy() first."
            )
        
        # Build workflow
        try:
            workflow = self.strategy.build_workflow(request, self.selected_models)
        except Exception as e:
            return WorkflowResult(success=False, error=f"Workflow build failed: {e}")
        
        # Execute
        return await self._execute_workflow(workflow)
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> WorkflowResult:
        """Submit and monitor workflow execution."""
        async with aiohttp.ClientSession() as session:
            # Queue
            try:
                async with session.post(
                    f"{self.url}/prompt",
                    json={"prompt": workflow},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    result = await resp.json()
                    
                    if "error" in result:
                        error_msg = result["error"].get("message", str(result["error"]))
                        for observer in self.observers:
                            observer.on_error("", error_msg)
                        return WorkflowResult(success=False, error=error_msg)
                    
                    prompt_id = result["prompt_id"]
                    for observer in self.observers:
                        observer.on_queued(prompt_id)
            
            except Exception as e:
                return WorkflowResult(success=False, error=f"Queue failed: {e}")
            
            # Poll for completion
            import time
            start_time = time.time()
            
            for _ in range(180):  # 6 minute timeout
                await asyncio.sleep(2)
                
                try:
                    async with session.get(f"{self.url}/history/{prompt_id}") as resp:
                        history = await resp.json()
                        
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            
                            if status.get("completed"):
                                # Extract outputs
                                outputs = []
                                for node_id, node_output in history[prompt_id].get("outputs", {}).items():
                                    if "images" in node_output:
                                        for img in node_output["images"]:
                                            filename = img["filename"]
                                            outputs.append(f"{self.url}/view?filename={filename}")
                                
                                result = WorkflowResult(
                                    success=True,
                                    outputs=outputs,
                                    execution_time=time.time() - start_time,
                                )
                                for observer in self.observers:
                                    observer.on_complete(prompt_id, result)
                                return result
                            
                            if status.get("status_str") == "error":
                                messages = status.get("messages", [])
                                error = "Execution failed"
                                for msg in messages:
                                    if msg[0] == "execution_error":
                                        error = msg[1].get("exception_message", error)[:500]
                                        break
                                
                                for observer in self.observers:
                                    observer.on_error(prompt_id, error)
                                return WorkflowResult(success=False, error=error)
                
                except Exception as e:
                    continue  # Retry on transient errors
            
            return WorkflowResult(success=False, error="Timeout waiting for completion")
    
    def add_observer(self, observer: ExecutionObserver) -> None:
        """Add an execution observer."""
        self.observers.append(observer)
    
    def status(self) -> str:
        """Pretty-print current status."""
        lines = ["=" * 50, "WORKFLOW EXECUTOR STATUS", "=" * 50, ""]
        
        if not self._discovered:
            lines.append("Status: Not discovered yet")
            return "\n".join(lines)
        
        hw = self.discovery.hardware
        lines.append(f"Target: {self.url}")
        lines.append(f"GPU: {hw.gpu_name}" if hw else "GPU: Unknown")
        if hw:
            lines.append(f"VRAM: {hw.vram_total_gb:.1f}GB total (constraint), {hw.vram_free_gb:.1f}GB currently free")
            lines.append(f"  -> Using {hw.vram_total_gb:.1f}GB as limit (models can be unloaded)")
        lines.append("")
        
        lines.append("Available Capabilities:")
        for cap in Capability:
            models = self.discovery.capabilities.get(cap, [])
            if models:
                lines.append(f"  {cap.value}: {len(models)} models")
        lines.append("")
        
        if self.strategy:
            lines.append(f"Selected Strategy: {self.strategy.name}")
            lines.append("Selected Models:")
            for role, model in self.selected_models.items():
                lines.append(f"  {role}: {model.filename}")
        else:
            lines.append("Strategy: Not selected")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def generate_video(
    comfyui_url: str,
    prompt: str,
    negative_prompt: str = "blurry, static, low quality",
    frames: int = 25,
    preference: str = "quality",
) -> WorkflowResult:
    """High-level convenience function for video generation.
    
    Handles discovery, strategy selection, and execution automatically.
    
    Args:
        comfyui_url: ComfyUI server URL
        prompt: Text description of desired video
        negative_prompt: What to avoid
        frames: Number of frames
        preference: "quality", "speed", or "balanced"
        
    Returns:
        WorkflowResult with success status and output URLs
    """
    executor = WorkflowExecutor(comfyui_url)
    
    print("Discovering capabilities...")
    await executor.discover()
    
    print("Selecting strategy...")
    strategy = executor.select_strategy(
        capability=Capability.TEXT_TO_VIDEO,
        preference=preference,
    )
    
    if not strategy:
        return WorkflowResult(
            success=False,
            error="No video generation strategy available for this system"
        )
    
    print(f"Generating with {strategy.name}...")
    request = VideoRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        frames=frames,
    )
    
    return await executor.execute(request)
