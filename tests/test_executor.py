"""Tests for workflow execution engine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import WorkflowExecutor, ExecutionStatus, ExecutionResult


@pytest.mark.asyncio
class TestWorkflowExecutor:
    """Test workflow execution engine functionality."""

    async def test_basic_node_execution(self):
        """Test execution of a single WorkflowNode."""

        executor = WorkflowExecutor()

        node = WorkflowNode(
            id="test-001",
            type=WorkflowNodeType.GENERATE,
            description="Test generation task"
        )

        vibe = Vibe(description="Test vibe")
        plan = WorkflowPlan(
            id="plan-001",
            source_vibe=vibe,
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        assert result.plan_id == "plan-001"
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 1
        assert result.node_results["test-001"].status == ExecutionStatus.COMPLETED
        assert "Generated content" in result.node_results["test-001"].result["result"]

    async def test_hierarchical_execution(self):
        """Test execution of hierarchical workflow with children."""

        executor = WorkflowExecutor()

        child1 = WorkflowNode(
            id="child-001",
            type=WorkflowNodeType.ANALYZE,
            description="Analyze requirements"
        )

        child2 = WorkflowNode(
            id="child-002",
            type=WorkflowNodeType.GENERATE,
            description="Generate based on analysis",
            dependencies=["child-001"]
        )

        parent = WorkflowNode(
            id="parent-001",
            type=WorkflowNodeType.COMPOSITE,
            description="Complete workflow",
            children=[child1, child2]
        )

        vibe = Vibe(description="Test hierarchical vibe")
        plan = WorkflowPlan(
            id="plan-hierarchical",
            source_vibe=vibe,
            root_nodes=[parent]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3  # parent + 2 children
        assert result.node_results["parent-001"].status == ExecutionStatus.COMPLETED
        assert result.node_results["child-001"].status == ExecutionStatus.COMPLETED
        assert result.node_results["child-002"].status == ExecutionStatus.COMPLETED

    async def test_dependency_handling(self):
        """Test that dependencies are respected in execution order."""

        executor = WorkflowExecutor()

        # Create nodes with dependencies
        node1 = WorkflowNode(
            id="step-001",
            type=WorkflowNodeType.ANALYZE,
            description="First step"
        )

        node2 = WorkflowNode(
            id="step-002",
            type=WorkflowNodeType.GENERATE,
            description="Second step depends on first",
            dependencies=["step-001"]
        )

        vibe = Vibe(description="Test dependency vibe")
        plan = WorkflowPlan(
            id="plan-deps",
            source_vibe=vibe,
            root_nodes=[node1, node2]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 2

        # Check that both nodes completed
        assert result.node_results["step-001"].status == ExecutionStatus.COMPLETED
        assert result.node_results["step-002"].status == ExecutionStatus.COMPLETED

        # Verify dependency was satisfied (timing-based, approximate)
        step1_started = result.node_results["step-001"].started_at
        step2_started = result.node_results["step-002"].started_at
        assert step2_started >= step1_started

    async def test_execution_failure_handling(self):
        """Test handling of node execution failures."""

        executor = WorkflowExecutor()

        # Override handler to simulate failure
        original_handler = executor._execute_generate

        async def failing_handler(node):
            if node.id == "fail-node":
                raise RuntimeError("Simulated failure")
            return await original_handler(node)

        # Replace in the handlers dict
        executor.node_handlers[WorkflowNodeType.GENERATE] = failing_handler

        failing_node = WorkflowNode(
            id="fail-node",
            type=WorkflowNodeType.GENERATE,
            description="This node will fail"
        )

        vibe = Vibe(description="Test failure handling")
        plan = WorkflowPlan(
            id="plan-failure",
            source_vibe=vibe,
            root_nodes=[failing_node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.FAILED
        assert result.node_results["fail-node"].status == ExecutionStatus.FAILED
        assert "Simulated failure" in result.node_results["fail-node"].error


@pytest.mark.asyncio
class TestEndToEndExecution:
    """Test complete end-to-end execution flow."""

    @patch('vibe_aigc.planner.LLMClient')
    async def test_complete_vibe_execution(self, mock_llm_client):
        """Test complete flow from Vibe to execution results."""

        from vibe_aigc.planner import MetaPlanner

        # Mock LLM response
        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "id": "plan-e2e-001",
            "root_nodes": [
                {
                    "id": "analyze-scene",
                    "type": "analyze",
                    "description": "Analyze scene requirements",
                    "parameters": {"detail_level": "high"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 30
                },
                {
                    "id": "generate-scene",
                    "type": "generate",
                    "description": "Generate cinematic scene",
                    "parameters": {"style": "dark, atmospheric"},
                    "dependencies": ["analyze-scene"],
                    "children": [],
                    "estimated_duration": 60
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        # Test complete execution
        planner = MetaPlanner()
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        result = await planner.execute(vibe)

        assert result["status"] == "completed"
        assert result["plan_id"] == "plan-e2e-001"
        assert result["vibe_description"] == "Create a cinematic sci-fi scene"

        execution_summary = result["execution_summary"]
        assert execution_summary["total_nodes"] == 2
        assert execution_summary["completed"] == 2
        assert execution_summary["failed"] == 0

        node_results = result["node_results"]
        assert "analyze-scene" in node_results
        assert "generate-scene" in node_results
        assert node_results["analyze-scene"]["status"] == "completed"
        assert node_results["generate-scene"]["status"] == "completed"