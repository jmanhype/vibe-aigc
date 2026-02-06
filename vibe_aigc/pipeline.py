"""
Pipeline: Workflow Chaining for Multi-Step Content Generation.

Enables chaining of tools into sequential pipelines where the output
of one step becomes the input for the next. This is the operational
layer that executes decomposed workflows from MetaPlanner.

Example:
    pipeline = Pipeline([
        PipelineStep(tool="image_generation", config={"width": 768}),
        PipelineStep(tool="upscale", config={"scale": 2}),
        PipelineStep(tool="video_generation", config={"frames": 33})
    ])
    result = await pipeline.execute({"prompt": "samurai in rain"})
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import asyncio
import logging

if TYPE_CHECKING:
    from .tools import ToolRegistry, BaseTool

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PipelineStep:
    """
    A single step in a pipeline.
    
    Attributes:
        tool: Tool name (resolved from registry) or BaseTool instance
        config: Configuration to merge with input for this step
        name: Optional human-readable name for this step
        condition: Optional condition function that determines if step should run
        on_error: Error handling strategy ('fail', 'skip', 'retry')
        max_retries: Number of retries if on_error='retry'
        output_key: Key to store this step's output in accumulated results
    """
    tool: Union[str, 'BaseTool']
    config: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    condition: Optional[callable] = None
    on_error: str = "fail"  # 'fail', 'skip', 'retry'
    max_retries: int = 3
    output_key: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            if isinstance(self.tool, str):
                self.name = self.tool
            else:
                self.name = getattr(self.tool, 'spec', None)
                if self.name:
                    self.name = self.name.name
                else:
                    self.name = "unknown_step"


@dataclass
class StepResult:
    """Result from executing a single pipeline step."""
    step_name: str
    step_index: int
    status: PipelineStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "step_index": self.step_index,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
            "retries": self.retries,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    status: PipelineStatus
    final_output: Any
    step_results: List[StepResult]
    total_duration: float
    started_at: str
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "final_output": self.final_output,
            "step_results": [r.to_dict() for r in self.step_results],
            "total_duration": self.total_duration,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata
        }
    
    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.COMPLETED
    
    def get_step_output(self, step_name_or_index: Union[str, int]) -> Optional[Any]:
        """Get output from a specific step."""
        if isinstance(step_name_or_index, int):
            if 0 <= step_name_or_index < len(self.step_results):
                return self.step_results[step_name_or_index].output
        else:
            for result in self.step_results:
                if result.step_name == step_name_or_index:
                    return result.output
        return None


class Pipeline:
    """
    Chains tools into sequential execution pipelines.
    
    The Pipeline class implements workflow chaining where:
    1. Each step receives the merged output of previous steps + its config
    2. Output keys allow accumulating results across steps
    3. Conditions can skip steps dynamically
    4. Error handling is configurable per step
    
    Example:
        # Create pipeline with tool names (resolved from registry)
        pipeline = Pipeline([
            PipelineStep(tool="image_generate", config={"size": "768x768"}),
            PipelineStep(tool="upscale", config={"scale": 2}),
            PipelineStep(tool="video_generate", config={"frames": 33})
        ], tool_registry=registry)
        
        result = await pipeline.execute({"prompt": "samurai in rain"})
        
        # Or create with tool instances directly
        pipeline = Pipeline([
            PipelineStep(tool=ImageGenerationTool(), config={"size": "1024x1024"}),
            PipelineStep(tool=UpscaleTool(), config={"scale": 4})
        ])
    """
    
    def __init__(
        self,
        steps: List[PipelineStep],
        tool_registry: Optional['ToolRegistry'] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize a pipeline.
        
        Args:
            steps: List of PipelineStep objects defining the chain
            tool_registry: Registry for resolving tool names to instances
            name: Optional name for this pipeline
            description: Optional description of what this pipeline does
        """
        self.steps = steps
        self.tool_registry = tool_registry
        self.name = name or f"pipeline_{len(steps)}_steps"
        self.description = description
        self._resolved_tools: Dict[int, 'BaseTool'] = {}
    
    def _resolve_tool(self, step: PipelineStep, step_index: int) -> 'BaseTool':
        """Resolve a tool from step definition."""
        # Check cache
        if step_index in self._resolved_tools:
            return self._resolved_tools[step_index]
        
        tool = step.tool
        
        # If already a tool instance, use it
        if not isinstance(tool, str):
            self._resolved_tools[step_index] = tool
            return tool
        
        # Resolve from registry
        if not self.tool_registry:
            raise ValueError(
                f"Step {step_index} uses tool name '{tool}' but no tool_registry provided. "
                "Either pass tool instances directly or provide a tool_registry."
            )
        
        resolved = self.tool_registry.get(tool)
        if not resolved:
            # Try common aliases
            aliases = {
                "image_generation": "image_generate",
                "video_generation": "video_generate",
                "audio_generation": "audio_generate",
                "text_generation": "llm_generate",
                "upscale": "image_upscale",
            }
            aliased = aliases.get(tool)
            if aliased:
                resolved = self.tool_registry.get(aliased)
        
        if not resolved:
            available = [t.name for t in self.tool_registry.list_tools()]
            raise ValueError(
                f"Tool '{tool}' not found in registry. Available tools: {available}"
            )
        
        self._resolved_tools[step_index] = resolved
        return resolved
    
    def _merge_inputs(
        self,
        accumulated: Dict[str, Any],
        step_config: Dict[str, Any],
        prev_output: Any
    ) -> Dict[str, Any]:
        """
        Merge accumulated results, previous output, and step config into inputs.
        
        Priority (highest to lowest):
        1. Step config (explicit configuration)
        2. Previous step output (chained results)
        3. Accumulated results (all previous outputs)
        """
        result = dict(accumulated)
        
        # Add previous output
        if isinstance(prev_output, dict):
            result.update(prev_output)
        elif prev_output is not None:
            result["_previous_output"] = prev_output
        
        # Step config has highest priority
        result.update(step_config)
        
        return result
    
    async def _execute_step(
        self,
        step: PipelineStep,
        step_index: int,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> StepResult:
        """Execute a single pipeline step with retry logic."""
        import time
        
        started_at = datetime.now().isoformat()
        start_time = time.time()
        retries = 0
        last_error = None
        
        # Check condition
        if step.condition:
            try:
                should_run = step.condition(inputs)
                if not should_run:
                    return StepResult(
                        step_name=step.name,
                        step_index=step_index,
                        status=PipelineStatus.COMPLETED,
                        output=inputs,  # Pass through unchanged
                        duration=0,
                        started_at=started_at,
                        completed_at=datetime.now().isoformat()
                    )
            except Exception as e:
                logger.warning(f"Condition check failed for step {step.name}: {e}")
        
        # Resolve tool
        try:
            tool = self._resolve_tool(step, step_index)
        except ValueError as e:
            return StepResult(
                step_name=step.name,
                step_index=step_index,
                status=PipelineStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
                started_at=started_at,
                completed_at=datetime.now().isoformat()
            )
        
        # Execute with retries
        max_attempts = step.max_retries + 1 if step.on_error == "retry" else 1
        
        for attempt in range(max_attempts):
            try:
                result = await tool.execute(inputs, context)
                
                if result.success:
                    output = result.output
                    # If output has a specific key we want, extract it
                    if step.output_key and isinstance(output, dict):
                        output = {step.output_key: output}
                    
                    return StepResult(
                        step_name=step.name,
                        step_index=step_index,
                        status=PipelineStatus.COMPLETED,
                        output=output,
                        duration=time.time() - start_time,
                        retries=retries,
                        started_at=started_at,
                        completed_at=datetime.now().isoformat()
                    )
                else:
                    last_error = result.error
                    retries += 1
                    
            except Exception as e:
                last_error = str(e)
                retries += 1
                logger.warning(f"Step {step.name} attempt {attempt + 1} failed: {e}")
            
            # Exponential backoff between retries
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        # All attempts failed
        if step.on_error == "skip":
            return StepResult(
                step_name=step.name,
                step_index=step_index,
                status=PipelineStatus.COMPLETED,
                output=inputs,  # Pass through unchanged
                error=f"Skipped after error: {last_error}",
                duration=time.time() - start_time,
                retries=retries,
                started_at=started_at,
                completed_at=datetime.now().isoformat()
            )
        
        return StepResult(
            step_name=step.name,
            step_index=step_index,
            status=PipelineStatus.FAILED,
            error=last_error,
            duration=time.time() - start_time,
            retries=retries,
            started_at=started_at,
            completed_at=datetime.now().isoformat()
        )
    
    async def execute(
        self,
        initial_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute the pipeline with the given initial input.
        
        Args:
            initial_input: Initial input dict (e.g., {"prompt": "..."})
            context: Optional execution context passed to all tools
        
        Returns:
            PipelineResult with final output and all step results
        """
        import time
        
        started_at = datetime.now().isoformat()
        start_time = time.time()
        
        step_results: List[StepResult] = []
        accumulated: Dict[str, Any] = dict(initial_input)
        current_output: Any = initial_input
        
        # Build context
        exec_context = context or {}
        exec_context["pipeline_name"] = self.name
        exec_context["total_steps"] = len(self.steps)
        
        for i, step in enumerate(self.steps):
            exec_context["current_step"] = i
            exec_context["step_name"] = step.name
            
            # Merge inputs for this step
            step_inputs = self._merge_inputs(accumulated, step.config, current_output)
            
            logger.info(f"Executing pipeline step {i + 1}/{len(self.steps)}: {step.name}")
            
            # Execute step
            step_result = await self._execute_step(step, i, step_inputs, exec_context)
            step_results.append(step_result)
            
            # Check for failure
            if step_result.status == PipelineStatus.FAILED:
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    final_output=None,
                    step_results=step_results,
                    total_duration=time.time() - start_time,
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    metadata={
                        "failed_step": i,
                        "failed_step_name": step.name,
                        "error": step_result.error
                    }
                )
            
            # Accumulate output
            current_output = step_result.output
            if isinstance(current_output, dict):
                accumulated.update(current_output)
        
        return PipelineResult(
            status=PipelineStatus.COMPLETED,
            final_output=current_output,
            step_results=step_results,
            total_duration=time.time() - start_time,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            metadata={"steps_executed": len(step_results)}
        )
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"Pipeline({self.name}, steps={step_names})"


class PipelineBuilder:
    """
    Fluent builder for creating pipelines.
    
    Example:
        pipeline = (PipelineBuilder("image_to_video")
            .add("image_generate", size="1024x1024")
            .add("upscale", scale=2)
            .add("video_generate", frames=33)
            .build(registry))
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description
        self._steps: List[PipelineStep] = []
    
    def add(
        self,
        tool: Union[str, 'BaseTool'],
        name: Optional[str] = None,
        on_error: str = "fail",
        condition: Optional[callable] = None,
        output_key: Optional[str] = None,
        **config
    ) -> 'PipelineBuilder':
        """Add a step to the pipeline."""
        step = PipelineStep(
            tool=tool,
            config=config,
            name=name,
            on_error=on_error,
            condition=condition,
            output_key=output_key
        )
        self._steps.append(step)
        return self
    
    def add_conditional(
        self,
        tool: Union[str, 'BaseTool'],
        condition: callable,
        **config
    ) -> 'PipelineBuilder':
        """Add a conditional step that only runs if condition returns True."""
        return self.add(tool, condition=condition, **config)
    
    def build(self, tool_registry: Optional['ToolRegistry'] = None) -> Pipeline:
        """Build the pipeline."""
        return Pipeline(
            steps=self._steps,
            tool_registry=tool_registry,
            name=self.name,
            description=self.description
        )


# Convenience factory functions

def create_image_pipeline(
    tool_registry: 'ToolRegistry',
    upscale: bool = False,
    upscale_factor: int = 2
) -> Pipeline:
    """Create a standard image generation pipeline."""
    steps = [
        PipelineStep(
            tool="image_generate",
            config={"size": "1024x1024"},
            name="generate_image"
        )
    ]
    
    if upscale:
        steps.append(PipelineStep(
            tool="image_upscale",
            config={"scale": upscale_factor},
            name="upscale_image",
            on_error="skip"  # Continue without upscaling if it fails
        ))
    
    return Pipeline(
        steps=steps,
        tool_registry=tool_registry,
        name="image_pipeline",
        description="Generate and optionally upscale images"
    )


def create_video_pipeline(
    tool_registry: 'ToolRegistry',
    generate_first_frame: bool = True,
    frames: int = 33
) -> Pipeline:
    """Create a standard video generation pipeline."""
    steps = []
    
    if generate_first_frame:
        steps.append(PipelineStep(
            tool="image_generate",
            config={"size": "768x768"},
            name="generate_first_frame"
        ))
    
    steps.append(PipelineStep(
        tool="video_generate",
        config={"frames": frames},
        name="generate_video"
    ))
    
    return Pipeline(
        steps=steps,
        tool_registry=tool_registry,
        name="video_pipeline",
        description="Generate video from prompt (optionally with generated first frame)"
    )
