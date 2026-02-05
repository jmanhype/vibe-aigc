"""Tests for adaptive replanning functionality."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.executor import ExecutionStatus


@pytest.mark.asyncio
class TestAdaptiveReplanning:
    """Test adaptive replanning and feedback integration."""

    @patch('vibe_aigc.planner.LLMClient')
    async def test_successful_execution_without_replanning(self, mock_llm_client):
        """Test that successful execution doesn't trigger replanning."""

        # Mock LLM client for planning
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "success-plan",
            "root_nodes": [
                {"id": "success-node", "type": "generate", "description": "Successful task",
                 "parameters": {}, "dependencies": [], "children": []}
            ]
        }
        mock_llm_client.return_value = mock_client

        planner = MetaPlanner()
        vibe = Vibe(description="Test successful execution")

        result = await planner.execute_with_adaptation(vibe)

        # Should complete successfully without replanning
        assert result["status"] == "completed"
        assert result["adaptation_info"]["total_attempts"] == 1
        assert len(result["adaptation_info"]["replan_history"]) == 0

        # Verify LLM was called only once
        assert mock_client.decompose_vibe.call_count == 1

    @patch('vibe_aigc.planner.LLMClient')
    async def test_adaptive_replanning_on_failure(self, mock_llm_client):
        """Test that failures trigger adaptive replanning."""

        # Mock LLM client for both initial and adaptive planning
        mock_client = AsyncMock()

        # First call returns plan that will fail
        # Second call returns adapted plan that succeeds
        mock_client.decompose_vibe.side_effect = [
            {  # Initial failing plan
                "id": "failing-plan",
                "root_nodes": [
                    {"id": "fail-node", "type": "generate", "description": "Will fail",
                     "parameters": {"should_fail": True}, "dependencies": [], "children": []}
                ]
            },
            {  # Adapted successful plan
                "id": "adapted-plan",
                "root_nodes": [
                    {"id": "success-node", "type": "generate", "description": "Will succeed",
                     "parameters": {"should_fail": False}, "dependencies": [], "children": []}
                ]
            }
        ]
        mock_llm_client.return_value = mock_client

        # Mock the executor to fail on first node, succeed on second
        planner = MetaPlanner()

        # Override node handler to simulate failure on specific nodes
        original_handler = planner.executor._execute_generate

        async def selective_failing_handler(node):
            if node.id == "fail-node":
                raise RuntimeError("Simulated failure for replanning test")
            return await original_handler(node)

        planner.executor.node_handlers[WorkflowNodeType.GENERATE] = selective_failing_handler

        vibe = Vibe(description="Test adaptive replanning")

        result = await planner.execute_with_adaptation(vibe)

        # Should eventually succeed after adaptation
        assert result["status"] == "completed"
        assert result["adaptation_info"]["total_attempts"] == 2  # Initial + adapted

        # Verify adaptation occurred
        adaptation_info = result["adaptation_info"]
        assert len(adaptation_info["adaptation_history"]) == 2
        assert len(adaptation_info["replan_history"]) > 0

        # Verify LLM was called twice (initial + adapted)
        assert mock_client.decompose_vibe.call_count == 2

    @patch('vibe_aigc.planner.LLMClient')
    async def test_max_replan_attempts_limit(self, mock_llm_client):
        """Test that replanning respects maximum attempt limits."""

        # Mock LLM to always return failing plans
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "always-failing-plan",
            "root_nodes": [
                {"id": "always-fail", "type": "generate", "description": "Always fails",
                 "parameters": {}, "dependencies": [], "children": []}
            ]
        }
        mock_llm_client.return_value = mock_client

        planner = MetaPlanner()
        planner.max_replan_attempts = 2  # Limit attempts for faster test

        # Mock handler to always fail
        async def always_failing_handler(node):
            raise RuntimeError("Always fails")

        planner.executor.node_handlers[WorkflowNodeType.GENERATE] = always_failing_handler

        vibe = Vibe(description="Test max replan limits")

        # Should raise RuntimeError after max attempts
        with pytest.raises(RuntimeError, match="Failed to execute vibe after 2 attempts"):
            await planner.execute_with_adaptation(vibe)

        # Should have attempted the maximum number of times
        assert mock_client.decompose_vibe.call_count == 2

    @patch('vibe_aigc.planner.LLMClient')
    async def test_vibe_adaptation_from_feedback(self, mock_llm_client):
        """Test vibe adaptation based on execution feedback."""

        mock_llm_client.return_value = AsyncMock()
        planner = MetaPlanner()

        # Create sample feedback indicating quality issues
        from vibe_aigc.executor import ExecutionResult, NodeResult
        execution_result = ExecutionResult("test-plan")
        execution_result.add_node_result(
            NodeResult("quality-node", ExecutionStatus.FAILED, error="Quality issues")
        )
        execution_result.add_feedback("quality-node", {
            "execution_quality": 0.2,  # Low quality
            "replan_indicators": ["low_quality_output", "execution_error"]
        })
        execution_result.suggest_replan({
            "node_id": "quality-node",
            "reason": "low_quality_output",
            "suggested_changes": ["enhance_output_detail", "use_higher_quality_approach"]
        })

        original_vibe = Vibe(description="Original task", constraints=["fast"])

        # Create a mock plan
        plan = WorkflowPlan(
            id="test-plan",
            source_vibe=original_vibe,
            root_nodes=[
                WorkflowNode(id="quality-node", type=WorkflowNodeType.GENERATE, description="Quality test")
            ]
        )

        adapted_vibe = await planner._adapt_vibe_from_feedback(original_vibe, execution_result, plan)

        # Should adapt the vibe based on feedback
        assert adapted_vibe.description != original_vibe.description
        assert "quality" in adapted_vibe.description.lower() or "detail" in adapted_vibe.description.lower()
        assert len(adapted_vibe.constraints) > len(original_vibe.constraints)

        # Should contain adaptation metadata
        assert "adaptation_reason" in adapted_vibe.metadata
        assert "original_description" in adapted_vibe.metadata
        assert adapted_vibe.metadata["original_description"] == original_vibe.description

    @patch('vibe_aigc.planner.LLMClient')
    async def test_error_based_adaptation(self, mock_llm_client):
        """Test vibe adaptation based on unexpected errors."""

        mock_llm_client.return_value = AsyncMock()
        planner = MetaPlanner()
        original_vibe = Vibe(description="Complex task", constraints=["detailed"])

        adapted_vibe = await planner._adapt_vibe_from_error(original_vibe, "TimeoutError: Operation timed out")

        # Should adapt for simpler approach
        assert "simplified" in adapted_vibe.description.lower()
        assert "use simpler approach" in adapted_vibe.constraints
        assert "avoid complex operations" in adapted_vibe.constraints

        # Should contain error metadata
        assert adapted_vibe.metadata["error_adaptation"] == True
        assert "TimeoutError" in adapted_vibe.metadata["original_error"]

    @patch('vibe_aigc.planner.LLMClient')
    async def test_adaptation_strategy_generation(self, mock_llm_client):
        """Test generation of adaptation strategies based on failure patterns."""

        mock_llm_client.return_value = AsyncMock()
        planner = MetaPlanner()

        # Test timeout failure pattern
        timeout_context = {
            "failed_nodes": [
                {"error": "TimeoutError: Operation timed out", "type": "generate", "quality": 0.0}
            ],
            "overall_quality": 0.0,
            "suggestions": [
                {"reason": "timeout", "suggested_changes": ["increase_timeout", "break_into_smaller_tasks"]}
            ]
        }

        original_vibe = Vibe(description="Complex task")
        strategy = planner._generate_adaptation_strategy(original_vibe, timeout_context)

        assert "shorter" in strategy["adapted_description"] or "simpler" in strategy["adapted_description"]
        assert "break complex tasks into smaller steps" in strategy["adapted_constraints"]
        assert "timeout_issues" in strategy["failure_patterns"]

        # Test memory failure pattern
        memory_context = {
            "failed_nodes": [
                {"error": "MemoryError: Out of memory", "type": "analyze", "quality": 0.0}
            ],
            "overall_quality": 0.0,
            "suggestions": []
        }

        memory_strategy = planner._generate_adaptation_strategy(original_vibe, memory_context)
        assert "minimize memory usage" in memory_strategy["adapted_constraints"]
        assert "memory_issues" in memory_strategy["failure_patterns"]

    @patch('vibe_aigc.planner.LLMClient')
    async def test_overall_quality_calculation(self, mock_llm_client):
        """Test calculation of overall execution quality."""

        mock_llm_client.return_value = AsyncMock()
        planner = MetaPlanner()

        # Create execution result with mixed quality feedback
        from vibe_aigc.executor import ExecutionResult
        execution_result = ExecutionResult("quality-test")

        execution_result.add_feedback("node1", {"execution_quality": 0.8})
        execution_result.add_feedback("node2", {"execution_quality": 0.6})
        execution_result.add_feedback("node3", {"execution_quality": 0.4})

        quality = planner._calculate_overall_quality(execution_result)
        expected_quality = (0.8 + 0.6 + 0.4) / 3
        assert abs(quality - expected_quality) < 0.01

        # Test empty feedback
        empty_result = ExecutionResult("empty-test")
        empty_quality = planner._calculate_overall_quality(empty_result)
        assert empty_quality == 0.0

    @patch('vibe_aigc.planner.LLMClient')
    async def test_adaptation_history_tracking(self, mock_llm_client):
        """Test that adaptation history is properly tracked."""

        mock_client = AsyncMock()

        # Use a more flexible approach that can handle variable number of calls
        plans = [
            {  # Initial plan
                "id": "plan-1",
                "root_nodes": [
                    {"id": "node-1", "type": "generate", "description": "Task 1",
                     "parameters": {}, "dependencies": [], "children": []}
                ]
            },
            {  # Adapted plan
                "id": "plan-2",
                "root_nodes": [
                    {"id": "node-2", "type": "generate", "description": "Task 2",
                     "parameters": {}, "dependencies": [], "children": []}
                ]
            }
        ]

        plan_call_count = 0
        async def plan_side_effect(vibe, knowledge_context=None, tools_context=None):
            nonlocal plan_call_count
            result = plans[min(plan_call_count, len(plans) - 1)]
            plan_call_count += 1
            return result

        mock_client.decompose_vibe.side_effect = plan_side_effect
        mock_llm_client.return_value = mock_client

        planner = MetaPlanner()

        # Mock to fail first, succeed second
        call_count = 0

        async def counting_handler(node):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First attempt fails")
            else:
                # Return high quality result to prevent further replanning
                return {
                    "result": "This is a very detailed and high-quality successful result that should meet quality standards"
                }

        planner.executor.node_handlers[WorkflowNodeType.GENERATE] = counting_handler

        vibe = Vibe(description="Test adaptation history")
        result = await planner.execute_with_adaptation(vibe)

        # Verify history tracking
        adaptation_info = result["adaptation_info"]

        assert len(adaptation_info["adaptation_history"]) == 2  # Two attempts
        assert len(adaptation_info["replan_history"]) == 1  # One adaptation

        # Verify history content
        first_attempt = adaptation_info["adaptation_history"][0]
        assert first_attempt["attempt"] == 1
        assert "feedback_data" in first_attempt
        assert "replan_suggestions" in first_attempt

        replan_record = adaptation_info["replan_history"][0]
        assert "timestamp" in replan_record
        assert "original_description" in replan_record
        assert "adaptation_strategy" in replan_record

    @patch('vibe_aigc.planner.LLMClient')
    async def test_node_type_lookup_in_plan(self, mock_llm_client):
        """Test looking up node types from plan structure."""

        mock_llm_client.return_value = AsyncMock()
        planner = MetaPlanner()

        # Create nested plan structure
        child = WorkflowNode(id="child-1", type=WorkflowNodeType.ANALYZE, description="Child task")
        parent = WorkflowNode(
            id="parent-1",
            type=WorkflowNodeType.COMPOSITE,
            description="Parent task",
            children=[child]
        )

        plan = WorkflowPlan(
            id="nested-plan",
            source_vibe=Vibe(description="Nested test"),
            root_nodes=[parent]
        )

        # Test lookup
        assert planner._get_node_type_from_plan(plan, "parent-1") == "composite"
        assert planner._get_node_type_from_plan(plan, "child-1") == "analyze"
        assert planner._get_node_type_from_plan(plan, "nonexistent") == "unknown"