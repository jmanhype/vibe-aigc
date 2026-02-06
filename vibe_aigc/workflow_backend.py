"""Workflow Backend — Video generation via WorkflowRegistry/Composer + ComfyPilot.

Replaces hardcoded workflows with the select-or-compose pattern.
Uses ComfyPilot (via ComfyUI Manager) for:
- Auto-downloading missing models
- Installing required custom nodes
- Self-upgrading capabilities
"""

import asyncio
import aiohttp
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from .workflow_registry import (
    WorkflowRegistry, 
    WorkflowRunner, 
    WorkflowCapability,
    create_registry
)
from .workflow_composer import WorkflowFactory, compose_wan_video
from .model_registry import ModelRegistry, ModelCapability


@dataclass
class GenerationResult:
    """Result of a video generation."""
    success: bool
    output_path: Optional[str] = None
    output_url: Optional[str] = None
    error: Optional[str] = None
    prompt_id: Optional[str] = None
    
    @property
    def images(self):
        """Compatibility with old interface."""
        if self.output_url:
            return [self.output_url]
        if self.output_path:
            return [self.output_path]
        return []


class WorkflowBackend:
    """Video generation backend using WorkflowRegistry/Composer + ComfyPilot.
    
    This is the unified backend that:
    1. Tries to SELECT a pre-made workflow from registry
    2. Falls back to COMPOSE a new workflow from atomic tools
    3. Uses ComfyPilot to auto-download missing models
    4. Runs the workflow via ComfyUI API
    
    ComfyPilot Integration (via ModelRegistry):
    - call_comfy_pilot() - interface to tools
    - download_recommended_model() - get missing models
    - auto_upgrade() - self-improve capabilities
    """
    
    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        workflow_dirs: Optional[list] = None
    ):
        self.comfyui_url = comfyui_url.rstrip('/')
        
        # Initialize registry
        self.registry = create_registry(
            workflow_dirs=workflow_dirs or ['./workflows'],
            comfyui_url=comfyui_url
        )
        
        # Model registry (for ComfyPilot integration)
        self.model_registry = ModelRegistry(comfyui_url=comfyui_url)
        
        # Workflow factory (select or compose)
        self.factory = WorkflowFactory(registry=self.registry)
        
        # Runner
        self.runner = WorkflowRunner(comfyui_url)
        
        # Discovered?
        self._discovered = False
        self._models_scanned = False
    
    async def initialize(self) -> None:
        """Discover available workflows and scan models."""
        # Discover workflows
        self.registry.discover()
        self._discovered = True
        
        # Scan available models (for ComfyPilot)
        await self.model_registry.refresh()
        self._models_scanned = True
        
        # Also try to capture current workflow from ComfyUI
        current = await self.registry.capture_current()
        if current:
            print(f"Captured current workflow: {current.name}")
    
    async def ensure_models(self, capability: str) -> bool:
        """Ensure required models are available, download if needed via ComfyPilot.
        
        This is the AUTONOMOUS capability from the paper.
        """
        cap_map = {
            "text_to_video": ModelCapability.TEXT_TO_VIDEO,
            "image_to_video": ModelCapability.TEXT_TO_VIDEO,
            "text_to_image": ModelCapability.TEXT_TO_IMAGE,
        }
        
        model_cap = cap_map.get(capability)
        if not model_cap:
            return True  # Unknown capability, assume OK
        
        # Check if we have a model for this
        best = self.model_registry.get_best_for(model_cap)
        if best:
            print(f"Have model for {capability}: {best.filename}")
            return True
        
        # No model - try to download via ComfyPilot
        print(f"No model for {capability}, attempting download via ComfyPilot...")
        result = await self.model_registry.download_recommended_model(model_cap)
        
        if "error" in result:
            print(f"Download failed: {result['error']}")
            return False
        
        print(f"Downloaded model for {capability}")
        return True
    
    async def auto_upgrade(self) -> None:
        """Automatically upgrade to better models via ComfyPilot.
        
        Paper: "The system can AUTONOMOUSLY upgrade itself"
        """
        results = await self.model_registry.auto_upgrade(max_downloads=1)
        for r in results:
            print(f"Auto-upgrade {r['model']}: {r['result']}")
    
    def status(self) -> str:
        """Get registry status."""
        return self.registry.status()
    
    async def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "blurry, static, low quality",
        width: int = 512,
        height: int = 512,
        frames: int = 24,
        steps: int = 6,
        cfg: float = 6.0,
        seed: Optional[int] = None,
        capability: str = "text_to_video",
        auto_download: bool = True
    ) -> GenerationResult:
        """Generate video using workflow registry/composer + ComfyPilot.
        
        1. Ensure required models exist (download via ComfyPilot if needed)
        2. Try to find a pre-made workflow for the capability
        3. If not found, compose one from atomic tools
        4. Parameterize and run
        """
        if not self._discovered:
            await self.initialize()
        
        # Ensure we have models (ComfyPilot auto-download)
        if auto_download:
            await self.ensure_models(capability)
        
        # Generate seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        # Try to get workflow from registry first
        cap = WorkflowCapability(capability)
        workflows = self.registry.get_for_capability(cap)
        
        if workflows:
            # Use pre-made workflow
            workflow_spec = workflows[0]
            print(f"Using workflow: {workflow_spec.name}")
            
            workflow = workflow_spec.parameterize(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                frames=frames,
                seed=seed,
                steps=steps,
                cfg=cfg
            )
        else:
            # Compose a new workflow
            print(f"No pre-made workflow for {capability}, composing...")
            
            # For video, we need model files - try to detect from registry
            workflow = compose_wan_video(
                model_file="Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
                clip_file="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                vae_file="wan_2.1_vae.safetensors",
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                frames=frames,
                steps=steps,
                cfg=cfg,
                seed=seed
            )
        
        # Execute the workflow
        return await self._execute_workflow(workflow)
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> GenerationResult:
        """Execute a workflow via ComfyUI API."""
        try:
            async with aiohttp.ClientSession() as session:
                # Queue the workflow
                async with session.post(
                    f"{self.comfyui_url}/prompt",
                    json={"prompt": workflow},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    result = await resp.json()
                    
                    if 'error' in result:
                        return GenerationResult(
                            success=False,
                            error=str(result['error'])
                        )
                    
                    prompt_id = result['prompt_id']
                
                # Wait for completion (10 min timeout for video)
                for _ in range(300):
                    await asyncio.sleep(2)
                    
                    async with session.get(
                        f"{self.comfyui_url}/history/{prompt_id}"
                    ) as resp:
                        history = await resp.json()
                        
                        if prompt_id in history:
                            status = history[prompt_id].get('status', {})
                            status_str = status.get('status_str', '')
                            
                            if status_str == 'success' or status.get('completed'):
                                # Find output
                                outputs = history[prompt_id].get('outputs', {})
                                for node_id, node_output in outputs.items():
                                    # Check for images/videos
                                    if 'images' in node_output:
                                        for img in node_output['images']:
                                            filename = img.get('filename', '')
                                            subfolder = img.get('subfolder', '')
                                            
                                            url = f"{self.comfyui_url}/view?"
                                            url += f"filename={filename}"
                                            if subfolder:
                                                url += f"&subfolder={subfolder}"
                                            
                                            return GenerationResult(
                                                success=True,
                                                output_url=url,
                                                prompt_id=prompt_id
                                            )
                                    
                                    # Check for gifs/videos (VHS nodes)
                                    if 'gifs' in node_output:
                                        for gif in node_output['gifs']:
                                            filename = gif.get('filename', '')
                                            return GenerationResult(
                                                success=True,
                                                output_url=f"{self.comfyui_url}/view?filename={filename}",
                                                prompt_id=prompt_id
                                            )
                                
                                # No output found but completed
                                return GenerationResult(
                                    success=True,
                                    prompt_id=prompt_id,
                                    error="Completed but no output found"
                                )
                            
                            elif status_str == 'error':
                                return GenerationResult(
                                    success=False,
                                    error="Workflow execution failed",
                                    prompt_id=prompt_id
                                )
                
                return GenerationResult(
                    success=False,
                    error="Timeout waiting for completion"
                )
                
        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e)
            )


# Factory function
def create_workflow_backend(
    comfyui_url: str = "http://127.0.0.1:8188",
    workflow_dirs: Optional[list] = None
) -> WorkflowBackend:
    """Create a workflow backend."""
    return WorkflowBackend(
        comfyui_url=comfyui_url,
        workflow_dirs=workflow_dirs
    )
