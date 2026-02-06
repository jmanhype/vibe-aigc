"""Workflow Composer — Build workflows from atomic tools.

Paper Section 5.4: "define data-flow topology"

Two modes:
1. SELECT pre-made workflows (WorkflowRegistry)
2. COMPOSE new workflows from atomic tools (this module)

Atomic tools are small, reusable building blocks:
- Single nodes (TextEncode, Sample, etc.)
- Node groups (ModelLoader + VAE as a unit)
- Mini-workflows (encode + sample + decode)

The Composer wires them together based on capability requirements.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod


class PortType(Enum):
    """Data types that flow between tools."""
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    CONDITIONING = "CONDITIONING"
    LATENT = "LATENT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    NOISE = "NOISE"
    SIGMAS = "SIGMAS"
    GUIDER = "GUIDER"
    SAMPLER = "SAMPLER"
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"


@dataclass
class Port:
    """Input or output port on an atomic tool."""
    name: str
    port_type: PortType
    optional: bool = False


@dataclass
class AtomicTool(ABC):
    """Base class for composable atomic tools.
    
    Each tool represents a capability that can be wired into a workflow.
    """
    name: str
    description: str = ""
    
    @property
    @abstractmethod
    def inputs(self) -> List[Port]:
        """What this tool needs."""
        pass
    
    @property
    @abstractmethod
    def outputs(self) -> List[Port]:
        """What this tool produces."""
        pass
    
    @abstractmethod
    def to_nodes(self, node_id_start: int, connections: Dict[str, Tuple[str, int]]) -> Tuple[Dict[str, Any], int]:
        """Convert to ComfyUI node(s).
        
        Args:
            node_id_start: First node ID to use
            connections: Map of input_name -> (source_node_id, source_slot)
            
        Returns:
            (nodes_dict, next_node_id)
        """
        pass


@dataclass 
class NodeTool(AtomicTool):
    """A single ComfyUI node as an atomic tool."""
    node_type: str = ""
    input_ports: List[Port] = field(default_factory=list)
    output_ports: List[Port] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def inputs(self) -> List[Port]:
        return self.input_ports
    
    @property
    def outputs(self) -> List[Port]:
        return self.output_ports
    
    def to_nodes(self, node_id_start: int, connections: Dict[str, Tuple[str, int]]) -> Tuple[Dict[str, Any], int]:
        node_id = str(node_id_start)
        
        inputs = dict(self.defaults)
        for port in self.input_ports:
            if port.name in connections:
                inputs[port.name] = list(connections[port.name])
        
        nodes = {
            node_id: {
                "class_type": self.node_type,
                "inputs": inputs
            }
        }
        
        return nodes, node_id_start + 1


# ============================================================================
# STANDARD ATOMIC TOOLS — The building blocks
# ============================================================================

class Tools:
    """Library of standard atomic tools."""
    
    @staticmethod
    def model_loader(model_type: str = "checkpoint") -> NodeTool:
        """Load a model (checkpoint, unet, or diffusion model)."""
        if model_type == "checkpoint":
            return NodeTool(
                name="model_loader",
                description="Load a checkpoint model",
                node_type="CheckpointLoaderSimple",
                input_ports=[],
                output_ports=[
                    Port("model", PortType.MODEL),
                    Port("clip", PortType.CLIP),
                    Port("vae", PortType.VAE),
                ],
                defaults={"ckpt_name": ""}
            )
        elif model_type == "unet":
            return NodeTool(
                name="unet_loader",
                description="Load a UNET/diffusion model",
                node_type="UNETLoader",
                input_ports=[],
                output_ports=[Port("model", PortType.MODEL)],
                defaults={"unet_name": "", "weight_dtype": "default"}
            )
        elif model_type == "diffusion_kj":
            return NodeTool(
                name="diffusion_loader_kj",
                description="Load diffusion model (KJNodes)",
                node_type="DiffusionModelLoaderKJ",
                input_ports=[],
                output_ports=[Port("model", PortType.MODEL)],
                defaults={"model_path": "", "weight_dtype": "default"}
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def clip_loader(clip_type: str = "default") -> NodeTool:
        """Load a CLIP/text encoder."""
        return NodeTool(
            name="clip_loader",
            description="Load CLIP text encoder",
            node_type="CLIPLoader",
            input_ports=[],
            output_ports=[Port("clip", PortType.CLIP)],
            defaults={"clip_name": "", "type": clip_type}
        )
    
    @staticmethod
    def vae_loader() -> NodeTool:
        """Load a VAE."""
        return NodeTool(
            name="vae_loader",
            description="Load VAE",
            node_type="VAELoader",
            input_ports=[],
            output_ports=[Port("vae", PortType.VAE)],
            defaults={"vae_name": ""}
        )
    
    @staticmethod
    def text_encode() -> NodeTool:
        """Encode text to conditioning."""
        return NodeTool(
            name="text_encode",
            description="Encode text prompt to conditioning",
            node_type="CLIPTextEncode",
            input_ports=[Port("clip", PortType.CLIP)],
            output_ports=[Port("conditioning", PortType.CONDITIONING)],
            defaults={"text": ""}
        )
    
    @staticmethod
    def empty_latent(for_video: bool = False) -> NodeTool:
        """Create empty latent image/video."""
        if for_video:
            return NodeTool(
                name="empty_latent_video",
                description="Create empty latent for video",
                node_type="EmptyLatentImage",
                input_ports=[],
                output_ports=[Port("latent", PortType.LATENT)],
                defaults={"width": 512, "height": 512, "batch_size": 24}
            )
        return NodeTool(
            name="empty_latent",
            description="Create empty latent image",
            node_type="EmptyLatentImage",
            input_ports=[],
            output_ports=[Port("latent", PortType.LATENT)],
            defaults={"width": 512, "height": 512, "batch_size": 1}
        )
    
    @staticmethod
    def ksampler() -> NodeTool:
        """Standard KSampler."""
        return NodeTool(
            name="ksampler",
            description="Sample latents with KSampler",
            node_type="KSampler",
            input_ports=[
                Port("model", PortType.MODEL),
                Port("positive", PortType.CONDITIONING),
                Port("negative", PortType.CONDITIONING),
                Port("latent_image", PortType.LATENT),
            ],
            output_ports=[Port("latent", PortType.LATENT)],
            defaults={
                "seed": 0,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            }
        )
    
    @staticmethod
    def sampler_custom_advanced() -> NodeTool:
        """Advanced sampler (for Wan, Flux, etc.)."""
        return NodeTool(
            name="sampler_advanced",
            description="Advanced sampler with separate noise/guider/sigmas",
            node_type="SamplerCustomAdvanced",
            input_ports=[
                Port("noise", PortType.NOISE),
                Port("guider", PortType.GUIDER),
                Port("sampler", PortType.SAMPLER),
                Port("sigmas", PortType.SIGMAS),
                Port("latent_image", PortType.LATENT),
            ],
            output_ports=[
                Port("output", PortType.LATENT),
                Port("denoised_output", PortType.LATENT),
            ]
        )
    
    @staticmethod
    def cfg_guider() -> NodeTool:
        """CFG Guider for advanced sampling."""
        return NodeTool(
            name="cfg_guider",
            description="Classifier-free guidance guider",
            node_type="CFGGuider",
            input_ports=[
                Port("model", PortType.MODEL),
                Port("positive", PortType.CONDITIONING),
                Port("negative", PortType.CONDITIONING),
            ],
            output_ports=[Port("guider", PortType.GUIDER)],
            defaults={"cfg": 6.0}
        )
    
    @staticmethod
    def basic_scheduler() -> NodeTool:
        """Basic scheduler for sigmas."""
        return NodeTool(
            name="basic_scheduler",
            description="Generate sigmas schedule",
            node_type="BasicScheduler",
            input_ports=[Port("model", PortType.MODEL)],
            output_ports=[Port("sigmas", PortType.SIGMAS)],
            defaults={"scheduler": "simple", "steps": 20, "denoise": 1.0}
        )
    
    @staticmethod
    def random_noise() -> NodeTool:
        """Random noise generator."""
        return NodeTool(
            name="random_noise",
            description="Generate random noise",
            node_type="RandomNoise",
            input_ports=[],
            output_ports=[Port("noise", PortType.NOISE)],
            defaults={"noise_seed": 0}
        )
    
    @staticmethod
    def ksampler_select() -> NodeTool:
        """Select sampler algorithm."""
        return NodeTool(
            name="ksampler_select",
            description="Select sampling algorithm",
            node_type="KSamplerSelect",
            input_ports=[],
            output_ports=[Port("sampler", PortType.SAMPLER)],
            defaults={"sampler_name": "euler"}
        )
    
    @staticmethod
    def vae_decode() -> NodeTool:
        """Decode latents to images."""
        return NodeTool(
            name="vae_decode",
            description="Decode latents to images",
            node_type="VAEDecode",
            input_ports=[
                Port("samples", PortType.LATENT),
                Port("vae", PortType.VAE),
            ],
            output_ports=[Port("image", PortType.IMAGE)]
        )
    
    @staticmethod
    def vae_encode() -> NodeTool:
        """Encode images to latents."""
        return NodeTool(
            name="vae_encode",
            description="Encode images to latents",
            node_type="VAEEncode",
            input_ports=[
                Port("pixels", PortType.IMAGE),
                Port("vae", PortType.VAE),
            ],
            output_ports=[Port("latent", PortType.LATENT)]
        )
    
    @staticmethod
    def load_image() -> NodeTool:
        """Load image from file."""
        return NodeTool(
            name="load_image",
            description="Load image from file",
            node_type="LoadImage",
            input_ports=[],
            output_ports=[
                Port("image", PortType.IMAGE),
                Port("mask", PortType.IMAGE),
            ],
            defaults={"image": ""}
        )
    
    @staticmethod
    def save_image() -> NodeTool:
        """Save images to file."""
        return NodeTool(
            name="save_image",
            description="Save images to file",
            node_type="SaveImage",
            input_ports=[Port("images", PortType.IMAGE)],
            output_ports=[],
            defaults={"filename_prefix": "vibe"}
        )
    
    @staticmethod
    def video_combine() -> NodeTool:
        """Combine images into video."""
        return NodeTool(
            name="video_combine",
            description="Combine images into video",
            node_type="VHS_VideoCombine",
            input_ports=[Port("images", PortType.IMAGE)],
            output_ports=[],
            defaults={
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "vibe",
                "format": "video/h264-mp4"
            }
        )


# ============================================================================
# WORKFLOW COMPOSER — Wire tools together
# ============================================================================

@dataclass
class WireSpec:
    """Specification for connecting tool outputs to inputs."""
    from_tool: str
    from_port: str
    to_tool: str
    to_port: str


@dataclass
class ToolInstance:
    """An instance of a tool with configuration."""
    tool: AtomicTool
    instance_id: str
    config: Dict[str, Any] = field(default_factory=dict)


class WorkflowComposer:
    """Compose workflows from atomic tools.
    
    Usage:
        composer = WorkflowComposer()
        
        # Add tools
        composer.add("loader", Tools.model_loader("checkpoint"), ckpt_name="model.safetensors")
        composer.add("pos_encode", Tools.text_encode(), text="a cat")
        composer.add("neg_encode", Tools.text_encode(), text="bad quality")
        composer.add("latent", Tools.empty_latent())
        composer.add("sampler", Tools.ksampler(), steps=20, cfg=7.0)
        composer.add("decode", Tools.vae_decode())
        composer.add("save", Tools.save_image())
        
        # Wire them together
        composer.wire("loader", "model", "sampler", "model")
        composer.wire("loader", "clip", "pos_encode", "clip")
        composer.wire("loader", "clip", "neg_encode", "clip")
        composer.wire("pos_encode", "conditioning", "sampler", "positive")
        composer.wire("neg_encode", "conditioning", "sampler", "negative")
        composer.wire("latent", "latent", "sampler", "latent_image")
        composer.wire("sampler", "latent", "decode", "samples")
        composer.wire("loader", "vae", "decode", "vae")
        composer.wire("decode", "image", "save", "images")
        
        # Build
        workflow = composer.build()
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolInstance] = {}
        self.wires: List[WireSpec] = []
    
    def add(self, instance_id: str, tool: AtomicTool, **config) -> "WorkflowComposer":
        """Add a tool instance."""
        self.tools[instance_id] = ToolInstance(
            tool=tool,
            instance_id=instance_id,
            config=config
        )
        return self
    
    def wire(self, from_tool: str, from_port: str, to_tool: str, to_port: str) -> "WorkflowComposer":
        """Connect an output port to an input port."""
        self.wires.append(WireSpec(from_tool, from_port, to_tool, to_port))
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the ComfyUI workflow."""
        # Assign node IDs
        tool_to_node_id: Dict[str, str] = {}
        current_id = 1
        
        for instance_id in self.tools:
            tool_to_node_id[instance_id] = str(current_id)
            current_id += 1
        
        # Build output port index (tool_id, port_name) -> (node_id, slot)
        output_slots: Dict[Tuple[str, str], Tuple[str, int]] = {}
        for instance_id, tool_instance in self.tools.items():
            node_id = tool_to_node_id[instance_id]
            for slot, port in enumerate(tool_instance.tool.outputs):
                output_slots[(instance_id, port.name)] = (node_id, slot)
        
        # Build connection map for each tool
        tool_connections: Dict[str, Dict[str, Tuple[str, int]]] = {
            tid: {} for tid in self.tools
        }
        
        for wire in self.wires:
            if (wire.from_tool, wire.from_port) in output_slots:
                source = output_slots[(wire.from_tool, wire.from_port)]
                tool_connections[wire.to_tool][wire.to_port] = source
        
        # Generate nodes
        workflow = {}
        for instance_id, tool_instance in self.tools.items():
            # Merge defaults with config
            tool = tool_instance.tool
            if isinstance(tool, NodeTool):
                node_id = tool_to_node_id[instance_id]
                inputs = dict(tool.defaults)
                inputs.update(tool_instance.config)
                
                # Add connections
                for port_name, source in tool_connections[instance_id].items():
                    inputs[port_name] = list(source)
                
                workflow[node_id] = {
                    "class_type": tool.node_type,
                    "inputs": inputs
                }
        
        return workflow
    
    def validate(self) -> List[str]:
        """Check for issues in the workflow."""
        issues = []
        
        # Check all required inputs are wired
        wired_inputs: Dict[str, Set[str]] = {tid: set() for tid in self.tools}
        for wire in self.wires:
            wired_inputs[wire.to_tool].add(wire.to_port)
        
        for instance_id, tool_instance in self.tools.items():
            for port in tool_instance.tool.inputs:
                if not port.optional and port.name not in wired_inputs[instance_id]:
                    # Check if it's in config
                    if port.name not in tool_instance.config:
                        issues.append(f"{instance_id}: Missing required input '{port.name}'")
        
        return issues


# ============================================================================
# WORKFLOW TEMPLATES — Common patterns as composable functions
# ============================================================================

def compose_txt2img(
    model_file: str,
    prompt: str,
    negative_prompt: str = "bad quality, blurry",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = 0
) -> Dict[str, Any]:
    """Compose a text-to-image workflow."""
    c = WorkflowComposer()
    
    c.add("loader", Tools.model_loader("checkpoint"), ckpt_name=model_file)
    c.add("pos", Tools.text_encode(), text=prompt)
    c.add("neg", Tools.text_encode(), text=negative_prompt)
    c.add("latent", Tools.empty_latent(), width=width, height=height)
    c.add("sampler", Tools.ksampler(), seed=seed, steps=steps, cfg=cfg)
    c.add("decode", Tools.vae_decode())
    c.add("save", Tools.save_image())
    
    c.wire("loader", "clip", "pos", "clip")
    c.wire("loader", "clip", "neg", "clip")
    c.wire("loader", "model", "sampler", "model")
    c.wire("pos", "conditioning", "sampler", "positive")
    c.wire("neg", "conditioning", "sampler", "negative")
    c.wire("latent", "latent", "sampler", "latent_image")
    c.wire("sampler", "latent", "decode", "samples")
    c.wire("loader", "vae", "decode", "vae")
    c.wire("decode", "image", "save", "images")
    
    return c.build()


def compose_wan_video(
    model_file: str,
    clip_file: str,
    vae_file: str,
    prompt: str,
    negative_prompt: str = "bad quality, blurry",
    width: int = 512,
    height: int = 512,
    frames: int = 24,
    steps: int = 6,
    cfg: float = 6.0,
    seed: int = 0
) -> Dict[str, Any]:
    """Compose a Wan-style video workflow with advanced sampling."""
    c = WorkflowComposer()
    
    # Loaders
    c.add("model", Tools.model_loader("diffusion_kj"), model_path=model_file)
    c.add("clip", Tools.clip_loader("wan"), clip_name=clip_file)
    c.add("vae", Tools.vae_loader(), vae_name=vae_file)
    
    # Encoding
    c.add("pos", Tools.text_encode(), text=prompt)
    c.add("neg", Tools.text_encode(), text=negative_prompt)
    
    # Latent
    c.add("latent", Tools.empty_latent(for_video=True), 
          width=width, height=height, batch_size=frames)
    
    # Advanced sampling components
    c.add("noise", Tools.random_noise(), noise_seed=seed)
    c.add("scheduler", Tools.basic_scheduler(), steps=steps)
    c.add("guider", Tools.cfg_guider(), cfg=cfg)
    c.add("sampler_select", Tools.ksampler_select(), sampler_name="euler")
    c.add("sampler", Tools.sampler_custom_advanced())
    
    # Output
    c.add("decode", Tools.vae_decode())
    c.add("video", Tools.video_combine(), frame_rate=24)
    
    # Wiring
    c.wire("clip", "clip", "pos", "clip")
    c.wire("clip", "clip", "neg", "clip")
    c.wire("model", "model", "scheduler", "model")
    c.wire("model", "model", "guider", "model")
    c.wire("pos", "conditioning", "guider", "positive")
    c.wire("neg", "conditioning", "guider", "negative")
    c.wire("noise", "noise", "sampler", "noise")
    c.wire("guider", "guider", "sampler", "guider")
    c.wire("sampler_select", "sampler", "sampler", "sampler")
    c.wire("scheduler", "sigmas", "sampler", "sigmas")
    c.wire("latent", "latent", "sampler", "latent_image")
    c.wire("sampler", "output", "decode", "samples")
    c.wire("vae", "vae", "decode", "vae")
    c.wire("decode", "image", "video", "images")
    
    return c.build()


# ============================================================================
# UNIFIED INTERFACE — Select OR Compose
# ============================================================================

class WorkflowFactory:
    """Unified interface for getting workflows.
    
    Tries to SELECT from registry first, falls back to COMPOSE.
    """
    
    def __init__(self, registry=None):
        self.registry = registry
    
    def get_workflow(
        self,
        capability: str,
        **params
    ) -> Dict[str, Any]:
        """Get a workflow for a capability.
        
        First tries to find a pre-made workflow in the registry.
        If not found, composes one from atomic tools.
        """
        from .workflow_registry import WorkflowCapability
        
        cap = WorkflowCapability(capability)
        
        # Try registry first
        if self.registry:
            workflows = self.registry.get_for_capability(cap)
            if workflows:
                # Use the first matching workflow
                return workflows[0].parameterize(**params)
        
        # Fall back to composition
        if capability == "text_to_image":
            return compose_txt2img(**params)
        elif capability == "text_to_video":
            return compose_wan_video(**params)
        else:
            raise ValueError(f"Cannot compose workflow for: {capability}")
