"""
Tests for the Pipeline workflow chaining system.
"""

import asyncio
import pytest
from typing import Dict, Any, Optional

from vibe_aigc.pipeline import (
    Pipeline,
    PipelineStep,
    PipelineResult,
    PipelineStatus,
    PipelineBuilder,
    StepResult
)
from vibe_aigc.tools import BaseTool, ToolResult, ToolSpec, ToolCategory, ToolRegistry
from vibe_aigc.models import Vibe
from vibe_aigc.planner import MetaPlanner


# ==================== Mock Tools ====================

class MockImageTool(BaseTool):
    """Mock image generation tool for testing."""
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generate",
            description="Generate images (mock)",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {"prompt": {"type": "string"}, "size": {"type": "string"}}
            },
            output_schema={"type": "object"}
        )
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        # Simulate image generation
        await asyncio.sleep(0.01)
        return ToolResult(
            success=True,
            output={
                "url": f"https://example.com/image_{inputs.get('size', '1024x1024')}.png",
                "prompt": inputs["prompt"],
                "size": inputs.get("size", "1024x1024")
            },
            metadata={"model": "mock-dalle", "provider": "mock"}
        )


class MockUpscaleTool(BaseTool):
    """Mock upscale tool for testing."""
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_upscale",
            description="Upscale images (mock)",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}, "scale": {"type": "integer"}}
            },
            output_schema={"type": "object"}
        )
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        await asyncio.sleep(0.01)
        scale = inputs.get("scale", 2)
        original_url = inputs.get("url", "unknown")
        return ToolResult(
            success=True,
            output={
                "url": f"{original_url.replace('.png', '')}_upscaled_{scale}x.png",
                "scale": scale,
                "original_url": original_url
            },
            metadata={"upscaler": "mock-esrgan"}
        )


class MockVideoTool(BaseTool):
    """Mock video generation tool for testing."""
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_generate",
            description="Generate video (mock)",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {"prompt": {"type": "string"}, "frames": {"type": "integer"}}
            },
            output_schema={"type": "object"}
        )
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        await asyncio.sleep(0.02)
        return ToolResult(
            success=True,
            output={
                "url": "https://example.com/video.mp4",
                "frames": inputs.get("frames", 33),
                "prompt": inputs["prompt"]
            },
            metadata={"model": "mock-ltx"}
        )


class MockFailingTool(BaseTool):
    """Tool that fails for testing error handling."""
    
    def __init__(self, fail_count: int = 1):
        self.fail_count = fail_count
        self.attempts = 0
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="failing_tool",
            description="A tool that fails",
            category=ToolCategory.UTILITY,
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            return ToolResult(
                success=False,
                output=None,
                error=f"Simulated failure (attempt {self.attempts})"
            )
        return ToolResult(
            success=True,
            output={"status": "recovered", "attempts": self.attempts}
        )


# ==================== Fixtures ====================

@pytest.fixture
def mock_registry():
    """Create a registry with mock tools."""
    registry = ToolRegistry()
    registry.register(MockImageTool())
    registry.register(MockUpscaleTool())
    registry.register(MockVideoTool())
    return registry


# ==================== Tests ====================

class TestPipelineBasic:
    """Basic pipeline functionality tests."""
    
    @pytest.mark.asyncio
    async def test_simple_pipeline(self, mock_registry):
        """Test a simple single-step pipeline."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(tool="image_generate", config={"size": "768x768"})
            ],
            tool_registry=mock_registry
        )
        
        result = await pipeline.execute({"prompt": "a red apple"})
        
        assert result.status == PipelineStatus.COMPLETED
        assert result.success
        assert len(result.step_results) == 1
        assert "url" in result.final_output
        assert "768x768" in result.final_output["url"]
    
    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self, mock_registry):
        """Test chaining multiple steps."""
        pipeline = Pipeline(
            steps=[
                PipelineStep(tool="image_generate", config={"size": "768x768"}, name="gen_image"),
                PipelineStep(tool="image_upscale", config={"scale": 2}, name="upscale")
            ],
            tool_registry=mock_registry
        )
        
        result = await pipeline.execute({"prompt": "mountain landscape"})
        
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.step_results) == 2
        # Verify chaining: upscale step got the URL from image step
        assert "upscaled_2x" in result.final_output["url"]
        assert "original_url" in result.final_output
    
    @pytest.mark.asyncio
    async def test_image_upscale_chain(self, mock_registry):
        """Test the specific image->upscale chain from requirements."""
        pipeline = Pipeline([
            PipelineStep(tool="image_generate", config={"size": "1024x1024"}),
            PipelineStep(tool="image_upscale", config={"scale": 4})
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({"prompt": "samurai in rain"})
        
        assert result.success
        assert result.step_results[0].status == PipelineStatus.COMPLETED
        assert result.step_results[1].status == PipelineStatus.COMPLETED
        # Verify the chain worked
        final = result.final_output
        assert "upscaled_4x" in final["url"]
        print(f"Pipeline completed in {result.total_duration:.3f}s")


class TestPipelineErrorHandling:
    """Test error handling strategies."""
    
    @pytest.mark.asyncio
    async def test_fail_on_error(self, mock_registry):
        """Test default fail behavior."""
        failing_tool = MockFailingTool(fail_count=10)  # Always fails
        mock_registry.register(failing_tool)
        
        pipeline = Pipeline([
            PipelineStep(tool="image_generate"),
            PipelineStep(tool="failing_tool", on_error="fail")
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({"prompt": "test"})
        
        assert result.status == PipelineStatus.FAILED
        assert not result.success
        assert result.step_results[0].status == PipelineStatus.COMPLETED
        assert result.step_results[1].status == PipelineStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_skip_on_error(self, mock_registry):
        """Test skip behavior continues pipeline."""
        failing_tool = MockFailingTool(fail_count=10)
        mock_registry.register(failing_tool)
        
        pipeline = Pipeline([
            PipelineStep(tool="image_generate"),
            PipelineStep(tool="failing_tool", on_error="skip"),
            PipelineStep(tool="image_upscale", config={"scale": 2})
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({"prompt": "test"})
        
        assert result.status == PipelineStatus.COMPLETED
        assert result.success
        # All three steps ran (middle was skipped)
        assert len(result.step_results) == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_error(self, mock_registry):
        """Test retry behavior."""
        failing_tool = MockFailingTool(fail_count=2)  # Fails twice, then succeeds
        mock_registry.register(failing_tool)
        
        pipeline = Pipeline([
            PipelineStep(tool="failing_tool", on_error="retry", max_retries=3)
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({})
        
        assert result.status == PipelineStatus.COMPLETED
        assert result.step_results[0].retries == 2  # Took 2 retries to succeed


class TestPipelineBuilder:
    """Test fluent pipeline builder."""
    
    @pytest.mark.asyncio
    async def test_builder_pattern(self, mock_registry):
        """Test fluent builder creates working pipeline."""
        pipeline = (PipelineBuilder("test_pipeline")
            .add("image_generate", size="1024x1024")
            .add("image_upscale", scale=2)
            .add("video_generate", frames=33)
            .build(mock_registry))
        
        assert len(pipeline.steps) == 3
        assert pipeline.name == "test_pipeline"
        
        result = await pipeline.execute({"prompt": "epic scene"})
        assert result.success
        assert len(result.step_results) == 3


class TestMetaPlannerIntegration:
    """Test integration with MetaPlanner."""
    
    @pytest.mark.asyncio
    async def test_vibe_to_pipeline_image(self, mock_registry):
        """Test MetaPlanner converts image vibe to pipeline."""
        planner = MetaPlanner(tool_registry=mock_registry)
        
        vibe = Vibe(
            description="A beautiful sunset over the ocean",
            domain="art"
        )
        
        pipeline = planner.vibe_to_pipeline(vibe)
        
        assert len(pipeline.steps) >= 1
        assert pipeline.steps[0].name == "generate_image"
    
    @pytest.mark.asyncio
    async def test_vibe_to_pipeline_video(self, mock_registry):
        """Test MetaPlanner converts video vibe to pipeline."""
        planner = MetaPlanner(tool_registry=mock_registry)
        
        vibe = Vibe(
            description="Animate a samurai drawing sword in the rain",
            domain="video"
        )
        
        pipeline = planner.vibe_to_pipeline(vibe)
        
        # Should have image generation + video generation
        step_names = [s.name for s in pipeline.steps]
        assert "generate_first_frame" in step_names or "generate_video" in step_names
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_method(self, mock_registry):
        """Test MetaPlanner.execute_pipeline() works."""
        planner = MetaPlanner(tool_registry=mock_registry)
        
        vibe = Vibe(
            description="Create an upscaled HD image of a dragon",
            metadata={"upscale_factor": 2}
        )
        
        result = await planner.execute_pipeline(vibe)
        
        assert result.success
        assert len(result.step_results) >= 1


class TestPipelineResult:
    """Test PipelineResult functionality."""
    
    @pytest.mark.asyncio
    async def test_get_step_output_by_index(self, mock_registry):
        """Test retrieving step output by index."""
        pipeline = Pipeline([
            PipelineStep(tool="image_generate"),
            PipelineStep(tool="image_upscale")
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({"prompt": "test"})
        
        step0_output = result.get_step_output(0)
        step1_output = result.get_step_output(1)
        
        assert step0_output is not None
        assert step1_output is not None
        assert "url" in step0_output
    
    @pytest.mark.asyncio
    async def test_get_step_output_by_name(self, mock_registry):
        """Test retrieving step output by name."""
        pipeline = Pipeline([
            PipelineStep(tool="image_generate", name="gen"),
            PipelineStep(tool="image_upscale", name="upscale")
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({"prompt": "test"})
        
        gen_output = result.get_step_output("gen")
        assert gen_output is not None
        assert "url" in gen_output
    
    @pytest.mark.asyncio
    async def test_result_to_dict(self, mock_registry):
        """Test serialization to dict."""
        pipeline = Pipeline([
            PipelineStep(tool="image_generate")
        ], tool_registry=mock_registry)
        
        result = await pipeline.execute({"prompt": "test"})
        result_dict = result.to_dict()
        
        assert "status" in result_dict
        assert "step_results" in result_dict
        assert "total_duration" in result_dict


# ==================== Main ====================

if __name__ == "__main__":
    # Run a quick test
    async def main():
        registry = ToolRegistry()
        registry.register(MockImageTool())
        registry.register(MockUpscaleTool())
        registry.register(MockVideoTool())
        
        print("Testing image->upscale chain...")
        pipeline = Pipeline([
            PipelineStep(tool="image_generate", config={"size": "768x768"}),
            PipelineStep(tool="image_upscale", config={"scale": 2})
        ], tool_registry=registry)
        
        result = await pipeline.execute({"prompt": "samurai in rain"})
        
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.total_duration:.3f}s")
        print(f"Steps: {len(result.step_results)}")
        for i, step in enumerate(result.step_results):
            print(f"  [{i}] {step.step_name}: {step.status.value} ({step.duration:.3f}s)")
        print(f"Final output: {result.final_output}")
    
    asyncio.run(main())
