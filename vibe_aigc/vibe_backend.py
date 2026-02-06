"""Vibe Backend — The unified, general-purpose generation backend.

Paper-aligned architecture:
1. DISCOVER system capabilities via ComfyPilot
2. MATCH intent to available tools
3. COMPOSE workflows from discovered nodes
4. EXECUTE via ComfyUI
5. EVALUATE with VLM feedback
6. REFINE and retry if needed

This works with ANY ComfyUI setup — no hardcoded models or patterns.
"""

import asyncio
import aiohttp
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .discovery import (
    SystemDiscovery, 
    SystemCapabilities, 
    Capability,
    ModelSearch,
    discover_system
)
from .composer_general import GeneralComposer, create_composer
from .workflow_registry import WorkflowRegistry, WorkflowCapability, create_registry
from .vlm_feedback import VLMFeedback, create_vlm_feedback


@dataclass
class GenerationRequest:
    """A generation request."""
    prompt: str
    capability: Capability
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    frames: int = 24
    steps: int = 20
    cfg: float = 7.0
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result of a generation."""
    success: bool
    output_url: Optional[str] = None
    output_path: Optional[str] = None
    quality_score: float = 0.0
    feedback: Optional[str] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    prompt_improvements: List[str] = field(default_factory=list)
    error: Optional[str] = None
    workflow_used: Optional[str] = None
    model_used: Optional[str] = None
    attempts: int = 1


class VibeBackend:
    """The unified Vibe AIGC backend.
    
    This is the GENERAL implementation that:
    - Works with ANY ComfyUI setup
    - Discovers capabilities, doesn't assume them
    - Composes workflows from available nodes
    - Uses VLM feedback for quality control
    """
    
    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        workflow_dirs: Optional[List[str]] = None,
        enable_vlm: bool = True,
        max_attempts: int = 3,
        quality_threshold: float = 7.0
    ):
        self.url = comfyui_url.rstrip('/')
        self.max_attempts = max_attempts
        self.quality_threshold = quality_threshold
        
        # Core components
        self.discovery = SystemDiscovery(comfyui_url)
        self.registry = create_registry(
            workflow_dirs=workflow_dirs or ['./workflows'],
            comfyui_url=comfyui_url
        )
        self.vlm = create_vlm_feedback() if enable_vlm else None
        
        # State
        self.capabilities: Optional[SystemCapabilities] = None
        self.composer: Optional[GeneralComposer] = None
        self._initialized = False
    
    async def initialize(self) -> SystemCapabilities:
        """Initialize the backend — discover system capabilities."""
        print("Discovering system capabilities via ComfyPilot...")
        
        # Discover hardware, nodes, models
        self.capabilities = await self.discovery.discover()
        
        # Create composer from discovered capabilities
        self.composer = create_composer(self.capabilities)
        
        # Discover saved workflows
        self.registry.discover()
        
        self._initialized = True
        
        print(self.capabilities.summary())
        return self.capabilities
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate content based on request.
        
        Flow:
        1. Check if we have the capability
        2. Try saved workflow first
        3. Fall back to composed workflow
        4. Execute and evaluate
        5. Refine if needed
        """
        if not self._initialized:
            await self.initialize()
        
        # Check capability
        if not self.capabilities.has_capability(request.capability):
            # Check if we can recommend models to add
            search = ModelSearch(self.capabilities.hardware)
            recommendations = await search.recommend_for_capability(request.capability)
            
            if recommendations:
                rec_names = [r.get("name", "Unknown") for r in recommendations[:3]]
                return GenerationResult(
                    success=False,
                    error=f"Capability {request.capability.value} not available. "
                          f"Consider installing: {', '.join(rec_names)}"
                )
            else:
                return GenerationResult(
                    success=False,
                    error=f"Capability {request.capability.value} not available on this system."
                )
        
        # Generate seed if not provided
        if request.seed is None:
            import random
            request.seed = random.randint(0, 2**32 - 1)
        
        # Try to get workflow
        workflow = await self._get_workflow(request)
        if not workflow:
            return GenerationResult(
                success=False,
                error="Could not create workflow for this request"
            )
        
        # Execute with VLM feedback loop
        return await self._execute_with_feedback(request, workflow)
    
    async def _get_workflow(self, request: GenerationRequest) -> Optional[Dict[str, Any]]:
        """Get a workflow — saved or composed."""
        
        # Map Capability to WorkflowCapability
        cap_map = {
            Capability.TEXT_TO_IMAGE: WorkflowCapability.TEXT_TO_IMAGE,
            Capability.TEXT_TO_VIDEO: WorkflowCapability.TEXT_TO_VIDEO,
            Capability.IMAGE_TO_VIDEO: WorkflowCapability.IMAGE_TO_VIDEO,
        }
        
        wf_cap = cap_map.get(request.capability)
        
        # Try saved workflow first
        if wf_cap:
            saved = self.registry.get_for_capability(wf_cap)
            if saved:
                print(f"Using saved workflow: {saved[0].name}")
                return saved[0].parameterize(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    frames=request.frames,
                    steps=request.steps,
                    cfg=request.cfg,
                    seed=request.seed
                )
        
        # Compose from available nodes
        print(f"Composing workflow for {request.capability.value}...")
        
        # Build kwargs based on capability
        kwargs = {
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg": request.cfg,
            "seed": request.seed
        }
        
        # Add frames only for video capabilities
        if request.capability in [Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO]:
            kwargs["frames"] = request.frames
        
        return self.composer.compose_for_capability(
            capability=request.capability,
            prompt=request.prompt,
            **kwargs
        )
    
    async def _execute_with_feedback(
        self, 
        request: GenerationRequest,
        workflow: Dict[str, Any]
    ) -> GenerationResult:
        """Execute workflow with VLM feedback loop."""
        
        current_prompt = request.prompt
        best_result = None
        best_score = 0.0
        
        for attempt in range(self.max_attempts):
            print(f"Attempt {attempt + 1}/{self.max_attempts}...")
            
            # Update prompt in workflow
            for node in workflow.values():
                if isinstance(node, dict) and node.get("class_type") in ["CLIPTextEncode"]:
                    inputs = node.get("inputs", {})
                    if "text" in inputs and not inputs["text"].startswith(("bad", "blur", "ugly")):
                        inputs["text"] = current_prompt
            
            # Execute
            result = await self._execute_workflow(workflow)
            
            if not result.success:
                if attempt < self.max_attempts - 1:
                    continue
                return result
            
            # VLM feedback
            if self.vlm and self.vlm.available and result.output_url:
                # Download image for VLM analysis
                feedback = None
                temp_path = None
                try:
                    import tempfile
                    import os
                    async with aiohttp.ClientSession() as session:
                        async with session.get(result.output_url) as resp:
                            if resp.status == 200:
                                content = await resp.read()
                                # Save to temp file (won't auto-delete)
                                suffix = '.png' if 'png' in result.output_url else '.webp'
                                fd, temp_path = tempfile.mkstemp(suffix=suffix)
                                os.write(fd, content)
                                os.close(fd)
                                
                                feedback = self.vlm.analyze_media(
                                    Path(temp_path), 
                                    current_prompt
                                )
                except Exception as e:
                    print(f"VLM feedback failed: {e}")
                    feedback = None
                finally:
                    # Clean up temp file (ignore errors on Windows)
                    if temp_path:
                        try:
                            import os
                            os.unlink(temp_path)
                        except:
                            pass  # Windows file locking, will be cleaned up by OS
                
                if feedback:
                    result.quality_score = feedback.quality_score
                    result.feedback = feedback.description
                    result.strengths = feedback.strengths
                    result.weaknesses = feedback.weaknesses
                    result.prompt_improvements = feedback.prompt_improvements
                    
                    if feedback.quality_score > best_score:
                        best_score = feedback.quality_score
                        best_result = result
                    
                    if feedback.quality_score >= self.quality_threshold:
                        print(f"Quality threshold met: {feedback.quality_score}/10")
                        result.attempts = attempt + 1
                        return result
                    
                    # Refine prompt for next attempt
                    if attempt < self.max_attempts - 1:
                        current_prompt = self.vlm.suggest_improvements(feedback, current_prompt)
                        print(f"Refined prompt: {current_prompt[:50]}...")
                else:
                    # VLM failed, return successful result
                    result.attempts = attempt + 1
                    return result
            else:
                # No VLM configured, return first successful result
                result.attempts = attempt + 1
                return result
        
        # Return best result if we ran out of attempts
        if best_result:
            best_result.attempts = self.max_attempts
            return best_result
        
        return GenerationResult(
            success=False,
            error="Failed after all attempts",
            attempts=self.max_attempts
        )
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> GenerationResult:
        """Execute a workflow via ComfyUI API."""
        try:
            async with aiohttp.ClientSession() as session:
                # Queue
                async with session.post(
                    f"{self.url}/prompt",
                    json={"prompt": workflow},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    
                    if "error" in data:
                        return GenerationResult(
                            success=False,
                            error=str(data["error"])
                        )
                    
                    prompt_id = data["prompt_id"]
                
                # Wait for completion
                for _ in range(300):  # 10 min timeout
                    await asyncio.sleep(2)
                    
                    async with session.get(f"{self.url}/history/{prompt_id}") as resp:
                        history = await resp.json()
                        
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            
                            if status.get("completed") or status.get("status_str") == "success":
                                # Find output
                                outputs = history[prompt_id].get("outputs", {})
                                for node_output in outputs.values():
                                    if "images" in node_output:
                                        img = node_output["images"][0]
                                        filename = img.get("filename", "")
                                        subfolder = img.get("subfolder", "")
                                        
                                        url = f"{self.url}/view?filename={filename}"
                                        if subfolder:
                                            url += f"&subfolder={subfolder}"
                                        
                                        return GenerationResult(
                                            success=True,
                                            output_url=url,
                                            output_path=filename  # TODO: Download locally
                                        )
                                    
                                    if "gifs" in node_output:
                                        gif = node_output["gifs"][0]
                                        return GenerationResult(
                                            success=True,
                                            output_url=f"{self.url}/view?filename={gif['filename']}",
                                            output_path=gif["filename"]
                                        )
                                
                                return GenerationResult(
                                    success=True,
                                    error="Completed but no output found"
                                )
                            
                            if status.get("status_str") == "error":
                                return GenerationResult(
                                    success=False,
                                    error="Workflow execution failed"
                                )
                
                return GenerationResult(success=False, error="Timeout")
                
        except Exception as e:
            return GenerationResult(success=False, error=str(e))
    
    def status(self) -> str:
        """Get backend status."""
        if not self._initialized:
            return "Not initialized. Call initialize() first."
        
        return self.capabilities.summary()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_vibe_backend(
    comfyui_url: str = "http://127.0.0.1:8188",
    **kwargs
) -> VibeBackend:
    """Create and initialize a Vibe backend."""
    backend = VibeBackend(comfyui_url=comfyui_url, **kwargs)
    await backend.initialize()
    return backend


async def generate(
    prompt: str,
    capability: str = "text_to_image",
    comfyui_url: str = "http://127.0.0.1:8188",
    **kwargs
) -> GenerationResult:
    """Quick generation function."""
    backend = await create_vibe_backend(comfyui_url)
    
    request = GenerationRequest(
        prompt=prompt,
        capability=Capability(capability),
        **kwargs
    )
    
    return await backend.generate(request)
