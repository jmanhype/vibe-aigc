"""Workflow Registry — Workflows as First-Class Tools.

Based on Paper Section 5.4: Agentic Orchestration
"Traverse atomic tool library, select optimal ensemble of components"

Key Insight: ComfyUI workflows ARE the atomic tools.
- Each .json workflow is a reusable capability
- Registry discovers and catalogs available workflows
- MetaPlanner selects workflows based on intent
- Workflows are parameterized, not built from scratch
"""

import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class WorkflowCapability(Enum):
    """What a workflow can do."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    UPSCALE = "upscale"
    INPAINT = "inpaint"
    AUDIO = "audio"
    COMPOSITE = "composite"  # Multi-step workflow


@dataclass
class WorkflowSpec:
    """Specification for a discovered workflow."""
    name: str
    path: Path
    capabilities: List[WorkflowCapability]
    description: str = ""
    
    # Parameterizable inputs (what can be customized)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Required models (for compatibility checking)
    required_models: List[str] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    # The actual workflow JSON (loaded lazily)
    _workflow: Optional[Dict] = field(default=None, repr=False)
    
    def load(self) -> Dict[str, Any]:
        """Load the workflow JSON."""
        if self._workflow is None:
            with open(self.path, 'r') as f:
                self._workflow = json.load(f)
        return self._workflow
    
    def parameterize(self, **kwargs) -> Dict[str, Any]:
        """Create a parameterized copy of the workflow.
        
        Common parameters:
        - prompt: Text prompt for generation
        - negative_prompt: What to avoid
        - seed: Random seed
        - width, height: Output dimensions
        - steps: Sampling steps
        """
        workflow = self.load()
        
        # Deep copy to avoid modifying original
        import copy
        parameterized = copy.deepcopy(workflow)
        
        # Apply parameters based on workflow structure
        self._apply_parameters(parameterized, kwargs)
        
        return parameterized
    
    def _apply_parameters(self, workflow: Dict, params: Dict) -> None:
        """Apply parameters to workflow nodes."""
        
        # Handle both API format and graph format
        nodes = workflow.get('nodes', workflow)
        if isinstance(nodes, list):
            # Graph serialize format (from ComfyUI UI)
            self._apply_to_graph_format(workflow, params)
        else:
            # API format (prompt dict)
            self._apply_to_api_format(workflow, params)
    
    def _apply_to_api_format(self, workflow: Dict, params: Dict) -> None:
        """Apply params to API-format workflow."""
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
                
            class_type = node.get('class_type', '')
            inputs = node.get('inputs', {})
            
            # Text encoding nodes
            if class_type == 'CLIPTextEncode':
                if 'prompt' in params and 'text' in inputs:
                    # Check if this is positive or negative
                    if inputs.get('text', '').startswith(('bad', 'blur', 'low', 'ugly')):
                        if 'negative_prompt' in params:
                            inputs['text'] = params['negative_prompt']
                    else:
                        inputs['text'] = params['prompt']
            
            # Sampler nodes
            if 'Sampler' in class_type or class_type == 'KSampler':
                if 'seed' in params:
                    inputs['seed'] = params['seed']
                if 'steps' in params:
                    inputs['steps'] = params['steps']
                if 'cfg' in params:
                    inputs['cfg'] = params['cfg']
            
            # Latent/resolution nodes
            if 'Latent' in class_type or 'Empty' in class_type:
                if 'width' in params:
                    inputs['width'] = params['width']
                if 'height' in params:
                    inputs['height'] = params['height']
                if 'frames' in params and 'length' in inputs:
                    inputs['length'] = params['frames']
                if 'frames' in params and 'batch_size' in inputs:
                    inputs['batch_size'] = params['frames']
            
            # Noise nodes
            if 'Noise' in class_type:
                if 'seed' in params:
                    inputs['noise_seed'] = params['seed']
            
            # Scheduler nodes
            if 'Scheduler' in class_type:
                if 'steps' in params:
                    inputs['steps'] = params['steps']
    
    def _apply_to_graph_format(self, workflow: Dict, params: Dict) -> None:
        """Apply params to graph-serialize format."""
        for node in workflow.get('nodes', []):
            node_type = node.get('type', '')
            widgets = node.get('widgets_values', [])
            
            if not widgets:
                continue
            
            # Text encoding - find prompt widgets
            if node_type == 'CLIPTextEncode' and widgets:
                if 'prompt' in params:
                    # First CLIPTextEncode is usually positive
                    if not any(neg in str(widgets[0]).lower() for neg in ['bad', 'blur', 'ugly']):
                        widgets[0] = params['prompt']
                    elif 'negative_prompt' in params:
                        widgets[0] = params['negative_prompt']
            
            # Sampler widgets vary by node type
            if 'seed' in params and 'Noise' in node_type:
                # RandomNoise: [seed, mode]
                if len(widgets) > 0:
                    widgets[0] = params['seed']


class WorkflowRegistry:
    """Registry for discovering and managing workflow tools.
    
    Workflows are discovered from:
    1. Local workflow directory
    2. ComfyUI's workflow storage
    3. Currently loaded workflow (via ComfyPilot)
    """
    
    def __init__(
        self, 
        workflow_dirs: Optional[List[Path]] = None,
        comfyui_url: Optional[str] = None
    ):
        self.workflow_dirs = workflow_dirs or []
        self.comfyui_url = comfyui_url
        self.workflows: Dict[str, WorkflowSpec] = {}
        self.capabilities: Dict[WorkflowCapability, List[WorkflowSpec]] = {}
    
    def discover(self) -> None:
        """Discover all available workflows."""
        self.workflows.clear()
        self.capabilities.clear()
        
        # Discover from local directories
        for directory in self.workflow_dirs:
            self._discover_directory(directory)
        
        # Build capability index
        for workflow in self.workflows.values():
            for cap in workflow.capabilities:
                if cap not in self.capabilities:
                    self.capabilities[cap] = []
                self.capabilities[cap].append(workflow)
        
        print(f"Discovered {len(self.workflows)} workflows")
        for cap, workflows in self.capabilities.items():
            if workflows:
                print(f"  {cap.value}: {len(workflows)} workflows")
    
    def _discover_directory(self, directory: Path) -> None:
        """Discover workflows from a directory."""
        if not directory.exists():
            return
        
        for path in directory.glob('**/*.json'):
            try:
                spec = self._load_workflow_spec(path)
                if spec:
                    self.workflows[spec.name] = spec
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    
    def _load_workflow_spec(self, path: Path) -> Optional[WorkflowSpec]:
        """Load workflow spec from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a workflow file
        if not self._is_workflow(data):
            return None
        
        # Infer capabilities from workflow content
        capabilities = self._infer_capabilities(data)
        
        # Extract metadata if present
        metadata = data.get('_vibe_metadata', {})
        
        # Get required models
        required_models = self._extract_required_models(data)
        
        return WorkflowSpec(
            name=metadata.get('name', path.stem),
            path=path,
            capabilities=capabilities,
            description=metadata.get('description', ''),
            parameters=metadata.get('parameters', {}),
            required_models=required_models,
            author=metadata.get('author', ''),
            version=metadata.get('version', '1.0'),
            tags=metadata.get('tags', []),
            _workflow=data
        )
    
    def _is_workflow(self, data: Dict) -> bool:
        """Check if JSON is a ComfyUI workflow."""
        # Graph format has 'nodes' list
        if 'nodes' in data and isinstance(data['nodes'], list):
            return True
        # API format has class_type in values
        if any(isinstance(v, dict) and 'class_type' in v for v in data.values()):
            return True
        return False
    
    def _infer_capabilities(self, data: Dict) -> List[WorkflowCapability]:
        """Infer workflow capabilities from its nodes."""
        capabilities = set()
        
        # Collect all node types
        node_types = set()
        
        if 'nodes' in data:
            # Graph format
            for node in data['nodes']:
                node_types.add(node.get('type', '').lower())
        else:
            # API format
            for node in data.values():
                if isinstance(node, dict):
                    node_types.add(node.get('class_type', '').lower())
        
        # Infer from node types
        node_str = ' '.join(node_types)
        
        if any(x in node_str for x in ['wan', 'ltx', 'animatediff', 'video']):
            if 'i2v' in node_str or 'image' in node_str:
                capabilities.add(WorkflowCapability.IMAGE_TO_VIDEO)
            capabilities.add(WorkflowCapability.TEXT_TO_VIDEO)
        
        if any(x in node_str for x in ['ksampler', 'sampler']) and 'video' not in node_str:
            capabilities.add(WorkflowCapability.TEXT_TO_IMAGE)
        
        if 'upscale' in node_str or 'esrgan' in node_str:
            capabilities.add(WorkflowCapability.UPSCALE)
        
        if 'inpaint' in node_str:
            capabilities.add(WorkflowCapability.INPAINT)
        
        if 'img2img' in node_str or 'vaeencode' in node_str:
            capabilities.add(WorkflowCapability.IMAGE_TO_IMAGE)
        
        return list(capabilities) if capabilities else [WorkflowCapability.COMPOSITE]
    
    def _extract_required_models(self, data: Dict) -> List[str]:
        """Extract model filenames from workflow."""
        models = set()
        
        # Look for model-loading nodes
        loader_inputs = ['ckpt_name', 'unet_name', 'vae_name', 'clip_name', 'model_path']
        
        if 'nodes' in data:
            for node in data['nodes']:
                widgets = node.get('widgets_values', [])
                # widgets can be list or dict
                if isinstance(widgets, list) and len(widgets) > 0:
                    first = widgets[0]
                    if isinstance(first, str) and first.endswith(('.safetensors', '.ckpt', '.pth', '.gguf')):
                        models.add(first)
        else:
            for node in data.values():
                if isinstance(node, dict):
                    inputs = node.get('inputs', {})
                    for key in loader_inputs:
                        if key in inputs and isinstance(inputs[key], str):
                            models.add(inputs[key])
        
        return list(models)
    
    async def capture_current(self) -> Optional[WorkflowSpec]:
        """Capture the currently loaded workflow from ComfyUI (via ComfyPilot)."""
        if not self.comfyui_url:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.comfyui_url}/claude-code/workflow",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    
                    workflow = data.get('workflow')
                    if not workflow:
                        return None
                    
                    # Create spec from live workflow
                    capabilities = self._infer_capabilities(workflow)
                    required_models = self._extract_required_models(workflow)
                    
                    return WorkflowSpec(
                        name="current_workflow",
                        path=Path("live://comfyui"),
                        capabilities=capabilities,
                        description="Currently loaded workflow",
                        required_models=required_models,
                        _workflow=workflow
                    )
        except Exception as e:
            print(f"Could not capture current workflow: {e}")
            return None
    
    def get(self, name: str) -> Optional[WorkflowSpec]:
        """Get a workflow by name."""
        return self.workflows.get(name)
    
    def get_for_capability(
        self, 
        capability: WorkflowCapability,
        required_models: Optional[Set[str]] = None
    ) -> List[WorkflowSpec]:
        """Get workflows that provide a capability.
        
        Args:
            capability: What you need the workflow to do
            required_models: If provided, filter to workflows compatible with these models
        """
        workflows = self.capabilities.get(capability, [])
        
        if required_models:
            # Filter to workflows whose required models are available
            workflows = [
                w for w in workflows
                if all(m in required_models for m in w.required_models)
            ]
        
        return workflows
    
    def save_workflow(
        self,
        workflow: Dict[str, Any],
        name: str,
        capabilities: List[WorkflowCapability],
        description: str = "",
        directory: Optional[Path] = None
    ) -> WorkflowSpec:
        """Save a workflow to the registry.
        
        Use this to save the current ComfyUI workflow as a reusable tool.
        """
        if directory is None:
            directory = self.workflow_dirs[0] if self.workflow_dirs else Path('./workflows')
        
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        
        # Add vibe metadata
        workflow_with_meta = workflow.copy()
        workflow_with_meta['_vibe_metadata'] = {
            'name': name,
            'description': description,
            'capabilities': [c.value for c in capabilities],
            'version': '1.0'
        }
        
        with open(path, 'w') as f:
            json.dump(workflow_with_meta, f, indent=2)
        
        # Create and register spec
        spec = WorkflowSpec(
            name=name,
            path=path,
            capabilities=capabilities,
            description=description,
            _workflow=workflow_with_meta
        )
        
        self.workflows[name] = spec
        for cap in capabilities:
            if cap not in self.capabilities:
                self.capabilities[cap] = []
            self.capabilities[cap].append(spec)
        
        return spec
    
    def status(self) -> str:
        """Pretty-print registry status."""
        lines = [
            "=" * 50,
            "WORKFLOW REGISTRY",
            "=" * 50,
            "",
            f"Total Workflows: {len(self.workflows)}",
            ""
        ]
        
        for cap in WorkflowCapability:
            workflows = self.capabilities.get(cap, [])
            if workflows:
                lines.append(f"{cap.value}:")
                for w in workflows[:3]:
                    lines.append(f"  • {w.name}" + (f" - {w.description[:40]}" if w.description else ""))
                if len(workflows) > 3:
                    lines.append(f"  ... and {len(workflows) - 3} more")
                lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# WORKFLOW EXECUTOR — Run parameterized workflows
# ============================================================================

class WorkflowRunner:
    """Execute workflows against ComfyUI."""
    
    def __init__(self, comfyui_url: str):
        self.url = comfyui_url.rstrip('/')
    
    async def run(
        self, 
        workflow: WorkflowSpec,
        **params
    ) -> Dict[str, Any]:
        """Run a workflow with parameters.
        
        Args:
            workflow: The workflow spec to run
            **params: Parameters to inject (prompt, seed, etc.)
            
        Returns:
            Execution result with outputs
        """
        # Parameterize the workflow
        parameterized = workflow.parameterize(**params)
        
        # Convert to API format if needed
        api_workflow = self._to_api_format(parameterized)
        
        # Execute
        return await self._execute(api_workflow)
    
    def _to_api_format(self, workflow: Dict) -> Dict:
        """Convert graph format to API format if needed."""
        if 'nodes' not in workflow:
            # Already API format
            return workflow
        
        # Graph format needs conversion
        # This is complex - for now, use workflow_api if available
        if 'workflow_api' in workflow:
            return workflow['workflow_api']
        
        # Fallback: try to extract from graph
        # This is a simplified conversion
        api = {}
        for node in workflow.get('nodes', []):
            node_id = str(node.get('id'))
            api[node_id] = {
                'class_type': node.get('type'),
                'inputs': self._extract_inputs(node, workflow)
            }
        return api
    
    def _extract_inputs(self, node: Dict, workflow: Dict) -> Dict:
        """Extract inputs from a graph-format node."""
        inputs = {}
        
        # Widget values become direct inputs
        widgets = node.get('widgets_values', [])
        # This requires knowing the node's input order - simplified for now
        
        # Link-based inputs
        for inp in node.get('inputs', []):
            link_id = inp.get('link')
            if link_id:
                # Find the link
                for link in workflow.get('links', []):
                    if link[0] == link_id:
                        from_node = str(link[1])
                        from_slot = link[2]
                        inputs[inp['name']] = [from_node, from_slot]
                        break
        
        return inputs
    
    async def _execute(self, workflow: Dict) -> Dict[str, Any]:
        """Execute workflow via ComfyUI API."""
        async with aiohttp.ClientSession() as session:
            # Queue the workflow
            async with session.post(
                f"{self.url}/prompt",
                json={"prompt": workflow},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                result = await resp.json()
                
                if 'error' in result:
                    return {"success": False, "error": result['error']}
                
                prompt_id = result['prompt_id']
            
            # Wait for completion
            for _ in range(300):  # 10 min timeout
                await asyncio.sleep(2)
                
                async with session.get(f"{self.url}/history/{prompt_id}") as resp:
                    history = await resp.json()
                    
                    if prompt_id in history:
                        status = history[prompt_id].get('status', {})
                        
                        if status.get('completed'):
                            outputs = []
                            for node_output in history[prompt_id].get('outputs', {}).values():
                                if 'images' in node_output:
                                    for img in node_output['images']:
                                        outputs.append(f"{self.url}/view?filename={img['filename']}")
                            
                            return {
                                "success": True,
                                "prompt_id": prompt_id,
                                "outputs": outputs
                            }
                        
                        if status.get('status_str') == 'error':
                            return {
                                "success": False,
                                "error": "Execution failed",
                                "details": status
                            }
            
            return {"success": False, "error": "Timeout"}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_registry(
    workflow_dirs: Optional[List[str]] = None,
    comfyui_url: Optional[str] = None
) -> WorkflowRegistry:
    """Create a workflow registry with default directories."""
    dirs = []
    
    if workflow_dirs:
        dirs = [Path(d) for d in workflow_dirs]
    else:
        # Default locations
        dirs = [
            Path('./workflows'),
            Path('./vibe_aigc/workflows'),
        ]
    
    registry = WorkflowRegistry(
        workflow_dirs=dirs,
        comfyui_url=comfyui_url
    )
    
    return registry
