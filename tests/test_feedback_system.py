"""Tests for feedback data collection and analysis."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import WorkflowExecutor, ExecutionStatus


@pytest.mark.asyncio
class TestFeedbackDataCollection:
    """Test feedback collection from node execution."""

    async def test_feedback_collection_on_success(self):
        """Test that successful execution collects useful feedback."""
        executor = WorkflowExecutor()

        node = WorkflowNode(
            id="feedback-test",
            type=WorkflowNodeType.GENERATE,
            description="Test feedback collection",
            estimated_duration=2
        )
        plan = WorkflowPlan(
            id="feedback-plan",
            source_vibe=Vibe(description="Test feedback"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert "feedback-test" in result.feedback_data

        feedback = result.feedback_data["feedback-test"]

        # Verify feedback structure
        assert "execution_quality" in feedback
        assert "resource_usage" in feedback
        assert "performance_metrics" in feedback
        assert "replan_indicators" in feedback
        assert "optimization_suggestions" in feedback

        # Verify quality assessment
        assert 0.0 <= feedback["execution_quality"] <= 1.0

        # Verify performance metrics
        perf_metrics = feedback["performance_metrics"]
        assert "duration" in perf_metrics
        assert "expected_duration" in perf_metrics
        assert "efficiency" in perf_metrics
        assert perf_metrics["duration"] > 0

    async def test_feedback_collection_on_failure(self):
        """Test feedback collection when node execution fails."""
        executor = WorkflowExecutor()

        # Override handler to simulate failure
        original_handler = executor._execute_generate

        async def failing_handler(node):
            if node.id == "fail-node":
                raise RuntimeError("Simulated failure for feedback testing")
            return await original_handler(node)

        executor.node_handlers[WorkflowNodeType.GENERATE] = failing_handler

        failing_node = WorkflowNode(
            id="fail-node",
            type=WorkflowNodeType.GENERATE,
            description="This node will fail"
        )

        plan = WorkflowPlan(
            id="feedback-failure-plan",
            source_vibe=Vibe(description="Test failure feedback"),
            root_nodes=[failing_node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.FAILED
        assert "fail-node" in result.feedback_data

        feedback = result.feedback_data["fail-node"]

        # Verify error feedback structure
        assert feedback["execution_quality"] == 0.0
        assert feedback["error_type"] == "RuntimeError"
        assert feedback["error_message"] == "Simulated failure for feedback testing"
        assert "suggested_changes" in feedback
        assert "replan_reason" in feedback
        assert feedback["replannable"] in [True, False]

    async def test_quality_assessment(self):
        """Test quality assessment of execution results."""
        executor = WorkflowExecutor()

        # Test with different result types
        test_cases = [
            # (node_id, mock_result, expected_quality_range)
            ("quality-good", {"result": "This is a detailed, meaningful result with substantial content"}, (0.7, 1.0)),
            ("quality-poor", {"result": "short"}, (0.3, 0.8)),
            ("quality-error", {"result": "content", "error": "some error"}, (0.0, 0.3)),
            ("quality-empty", None, (0.0, 0.1))
        ]

        for node_id, mock_result, (min_quality, max_quality) in test_cases:
            # Mock the handler to return specific result
            async def custom_handler(node):
                return mock_result

            executor.node_handlers[WorkflowNodeType.GENERATE] = custom_handler

            node = WorkflowNode(id=node_id, type=WorkflowNodeType.GENERATE, description="Quality test")
            plan = WorkflowPlan(
                id=f"quality-plan-{node_id}",
                source_vibe=Vibe(description="Quality test"),
                root_nodes=[node]
            )

            result = await executor.execute_plan(plan)

            if mock_result is not None:
                assert result.status == ExecutionStatus.COMPLETED
                feedback = result.feedback_data[node_id]
                quality = feedback["execution_quality"]

                assert min_quality <= quality <= max_quality, \
                    f"Quality {quality} not in expected range [{min_quality}, {max_quality}] for {node_id}"

    async def test_replan_suggestion_triggers(self):
        """Test conditions that trigger replanning suggestions."""
        executor = WorkflowExecutor()

        # Mock handler that returns low-quality results
        async def low_quality_handler(node):
            return {"result": "bad", "error": "quality issues"}

        executor.node_handlers[WorkflowNodeType.GENERATE] = low_quality_handler

        node = WorkflowNode(
            id="replan-trigger",
            type=WorkflowNodeType.GENERATE,
            description="Node that should trigger replanning"
        )

        plan = WorkflowPlan(
            id="replan-test",
            source_vibe=Vibe(description="Test replan triggers"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        # Should have replan suggestions due to low quality
        assert len(result.replan_suggestions) > 0

        suggestion = result.replan_suggestions[0]
        assert "node_id" in suggestion
        assert "reason" in suggestion
        assert suggestion["node_id"] == "replan-trigger"

    async def test_performance_metrics_tracking(self):
        """Test performance metrics collection and efficiency calculation."""
        executor = WorkflowExecutor()

        # Create node with estimated duration
        node = WorkflowNode(
            id="perf-test",
            type=WorkflowNodeType.GENERATE,
            description="Performance test",
            estimated_duration=1  # 1 second estimate
        )

        plan = WorkflowPlan(
            id="performance-plan",
            source_vibe=Vibe(description="Performance test"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        feedback = result.feedback_data["perf-test"]
        perf_metrics = feedback["performance_metrics"]

        # Verify performance tracking
        assert perf_metrics["duration"] > 0
        assert perf_metrics["expected_duration"] == 1
        assert "efficiency" in perf_metrics

        # Efficiency should be reasonable (task should complete faster than 1s)
        assert perf_metrics["efficiency"] > 0

    async def test_optimization_suggestions(self):
        """Test generation of optimization suggestions."""
        executor = WorkflowExecutor()

        # Mock slow execution
        original_sleep_time = 0.2  # Normal mock execution time

        # Override to simulate slow execution
        async def slow_handler(node):
            await asyncio.sleep(0.4)  # Much longer than estimated
            return {"result": "slow execution result"}

        executor.node_handlers[WorkflowNodeType.ANALYZE] = slow_handler

        node = WorkflowNode(
            id="slow-node",
            type=WorkflowNodeType.ANALYZE,
            description="Slow execution test",
            estimated_duration=0  # No estimate, so should trigger optimization suggestion for tasks > 0.1s
        )

        plan = WorkflowPlan(
            id="optimization-test",
            source_vibe=Vibe(description="Test optimization suggestions"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        feedback = result.feedback_data["slow-node"]
        suggestions = feedback["optimization_suggestions"]

        # Should suggest optimizations for slow execution
        assert len(suggestions) > 0
        # Common suggestions for slow execution
        expected_suggestions = ["reduce_task_complexity", "parallelize_subtasks"]
        assert any(suggestion in suggestions for suggestion in expected_suggestions)

    async def test_error_recovery_suggestions(self):
        """Test error-specific recovery suggestions."""
        executor = WorkflowExecutor()

        # Test different error types
        error_test_cases = [
            ("timeout-error", TimeoutError("Operation timed out"), ["increase_timeout", "break_into_smaller_tasks"]),
            ("memory-error", MemoryError("Out of memory"), ["reduce_data_size", "use_streaming_approach"]),
            ("generic-error", ValueError("Some value error"), ["retry_with_different_parameters"])
        ]

        for node_id, test_error, expected_suggestions in error_test_cases:
            # Mock handler to raise specific error
            async def error_handler(node):
                raise test_error

            executor.node_handlers[WorkflowNodeType.TRANSFORM] = error_handler

            node = WorkflowNode(id=node_id, type=WorkflowNodeType.TRANSFORM, description="Error test")
            plan = WorkflowPlan(
                id=f"error-recovery-{node_id}",
                source_vibe=Vibe(description="Error recovery test"),
                root_nodes=[node]
            )

            result = await executor.execute_plan(plan)

            assert result.status == ExecutionStatus.FAILED
            feedback = result.feedback_data[node_id]

            suggested_changes = feedback["suggested_changes"]

            # Should have appropriate recovery suggestions
            assert len(suggested_changes) > 0
            for expected in expected_suggestions:
                assert expected in suggested_changes, \
                    f"Expected suggestion '{expected}' not found in {suggested_changes}"

    async def test_should_replan_logic(self):
        """Test the should_replan() decision logic."""
        executor = WorkflowExecutor()

        # Create plan with multiple failing nodes to trigger replanning
        node1 = WorkflowNode(id="fail-1", type=WorkflowNodeType.GENERATE, description="Fail 1")
        node2 = WorkflowNode(id="fail-2", type=WorkflowNodeType.GENERATE, description="Fail 2")

        # Mock handler to fail
        async def failing_handler(node):
            raise RuntimeError(f"Failure in {node.id}")

        executor.node_handlers[WorkflowNodeType.GENERATE] = failing_handler

        plan = WorkflowPlan(
            id="should-replan-test",
            source_vibe=Vibe(description="Test should_replan logic"),
            root_nodes=[node1, node2]
        )

        result = await executor.execute_plan(plan)

        # With multiple failures, should recommend replanning
        assert result.should_replan() == True

        # Should have feedback for both failed nodes
        assert "fail-1" in result.feedback_data
        assert "fail-2" in result.feedback_data

        # Should have replan suggestions
        assert len(result.replan_suggestions) > 0