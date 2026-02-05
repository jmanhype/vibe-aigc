"""Tests for parallel execution functionality."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import time

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import WorkflowExecutor, ExecutionStatus


@pytest.mark.asyncio
class TestParallelExecution:
    """Test parallel execution of independent workflow nodes."""

    async def test_independent_nodes_execute_in_parallel(self):
        """Test that nodes without dependencies execute concurrently."""
        executor = WorkflowExecutor()

        # Create nodes that simulate work duration
        node1 = WorkflowNode(id="parallel-1", type=WorkflowNodeType.GENERATE, description="Task 1")
        node2 = WorkflowNode(id="parallel-2", type=WorkflowNodeType.GENERATE, description="Task 2")
        node3 = WorkflowNode(id="parallel-3", type=WorkflowNodeType.GENERATE, description="Task 3")

        plan = WorkflowPlan(
            id="parallel-test",
            source_vibe=Vibe(description="Test parallel execution"),
            root_nodes=[node1, node2, node3]
        )

        start_time = time.time()
        result = await executor.execute_plan(plan)
        execution_time = time.time() - start_time

        # Verify all nodes completed successfully
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3
        assert all(r.status == ExecutionStatus.COMPLETED for r in result.node_results.values())

        # Verify parallel efficiency is tracked
        assert hasattr(result, 'parallel_efficiency')
        assert hasattr(result, 'execution_groups')

        # Should have one execution group with all three nodes (since they're independent)
        assert len(result.execution_groups) == 1
        assert len(result.execution_groups[0]) == 3

        # Execution time should be close to individual task duration, not sum of all
        # Since our mock tasks take ~0.2s each, parallel should be ~0.2s, sequential would be ~0.6s
        assert execution_time < 0.5  # Allow some overhead, but much less than sequential

    async def test_dependency_order_maintained_in_parallel(self):
        """Test that dependency order is respected even with parallel execution."""
        executor = WorkflowExecutor()

        # Create dependency chain: step1 -> step2 -> step3, plus independent step4
        step1 = WorkflowNode(id="step-1", type=WorkflowNodeType.ANALYZE, description="First")
        step2 = WorkflowNode(id="step-2", type=WorkflowNodeType.GENERATE,
                           description="Second", dependencies=["step-1"])
        step3 = WorkflowNode(id="step-3", type=WorkflowNodeType.VALIDATE,
                           description="Third", dependencies=["step-2"])
        step4 = WorkflowNode(id="step-4", type=WorkflowNodeType.GENERATE, description="Independent")

        plan = WorkflowPlan(
            id="dependency-parallel-test",
            source_vibe=Vibe(description="Test dependencies with parallel"),
            root_nodes=[step1, step2, step3, step4]
        )

        result = await executor.execute_plan(plan)

        # All should complete successfully
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 4

        # Should have multiple execution groups due to dependencies
        assert len(result.execution_groups) >= 2

        # step1 and step4 should be in first group (independent)
        first_group = result.execution_groups[0]
        assert "step-1" in first_group
        assert "step-4" in first_group

        # step2 should be in a later group (depends on step1)
        # step3 should be in an even later group (depends on step2)
        step2_group_idx = None
        step3_group_idx = None
        for i, group in enumerate(result.execution_groups):
            if "step-2" in group:
                step2_group_idx = i
            if "step-3" in group:
                step3_group_idx = i

        assert step2_group_idx is not None
        assert step3_group_idx is not None
        assert step2_group_idx > 0  # step2 not in first group
        assert step3_group_idx > step2_group_idx  # step3 after step2

    async def test_parallel_execution_groups_algorithm(self):
        """Test the parallel execution groups building algorithm directly."""
        executor = WorkflowExecutor()

        # Create a complex dependency scenario
        nodeA = WorkflowNode(id="A", type=WorkflowNodeType.ANALYZE, description="Task A")
        nodeB = WorkflowNode(id="B", type=WorkflowNodeType.GENERATE, description="Task B")  # Independent
        nodeC = WorkflowNode(id="C", type=WorkflowNodeType.TRANSFORM,
                           description="Task C", dependencies=["A"])
        nodeD = WorkflowNode(id="D", type=WorkflowNodeType.VALIDATE,
                           description="Task D", dependencies=["A", "B"])
        nodeE = WorkflowNode(id="E", type=WorkflowNodeType.GENERATE,
                           description="Task E", dependencies=["C", "D"])

        nodes = [nodeA, nodeB, nodeC, nodeD, nodeE]

        # Create empty result for dependency checking
        from vibe_aigc.executor import ExecutionResult
        result = ExecutionResult("test")

        groups = executor._build_parallel_execution_groups(nodes, result)

        # Expected grouping:
        # Group 1: A, B (no dependencies)
        # Group 2: C, D (depend on A and/or B)
        # Group 3: E (depends on C, D)

        assert len(groups) == 3

        # First group should have A and B
        first_group_ids = [node.id for node in groups[0]]
        assert "A" in first_group_ids
        assert "B" in first_group_ids

        # Second group should have C and D
        second_group_ids = [node.id for node in groups[1]]
        assert "C" in second_group_ids
        assert "D" in second_group_ids

        # Third group should have E
        third_group_ids = [node.id for node in groups[2]]
        assert "E" in third_group_ids

    async def test_parallel_efficiency_calculation(self):
        """Test parallel efficiency calculation."""
        executor = WorkflowExecutor()

        # Create three independent tasks
        node1 = WorkflowNode(id="eff-1", type=WorkflowNodeType.GENERATE, description="Efficiency test 1")
        node2 = WorkflowNode(id="eff-2", type=WorkflowNodeType.GENERATE, description="Efficiency test 2")
        node3 = WorkflowNode(id="eff-3", type=WorkflowNodeType.GENERATE, description="Efficiency test 3")

        plan = WorkflowPlan(
            id="efficiency-test",
            source_vibe=Vibe(description="Test efficiency calculation"),
            root_nodes=[node1, node2, node3]
        )

        result = await executor.execute_plan(plan)

        # Verify efficiency calculation
        assert result.parallel_efficiency >= 0.0
        assert result.parallel_efficiency <= 1.0

        # With three parallel tasks, we should see some efficiency gain
        # (since actual duration < sum of individual durations)
        total_node_duration = sum(r.duration for r in result.node_results.values())
        assert result.total_duration < total_node_duration  # Parallel execution is faster

    async def test_hierarchical_parallel_execution(self):
        """Test parallel execution within hierarchical structures."""
        executor = WorkflowExecutor()

        # Create hierarchical structure with parallel children
        child1 = WorkflowNode(id="child-1", type=WorkflowNodeType.ANALYZE, description="Child 1")
        child2 = WorkflowNode(id="child-2", type=WorkflowNodeType.GENERATE, description="Child 2")
        child3 = WorkflowNode(id="child-3", type=WorkflowNodeType.TRANSFORM, description="Child 3")

        parent = WorkflowNode(
            id="parent",
            type=WorkflowNodeType.COMPOSITE,
            description="Parent with parallel children",
            children=[child1, child2, child3]
        )

        plan = WorkflowPlan(
            id="hierarchical-parallel-test",
            source_vibe=Vibe(description="Test hierarchical parallel execution"),
            root_nodes=[parent]
        )

        result = await executor.execute_plan(plan)

        # All nodes should complete successfully
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 4  # parent + 3 children

        # Verify all nodes completed
        for node_id in ["parent", "child-1", "child-2", "child-3"]:
            assert node_id in result.node_results
            assert result.node_results[node_id].status == ExecutionStatus.COMPLETED

    async def test_empty_plan_parallel_execution(self):
        """Test parallel execution with empty plan."""
        executor = WorkflowExecutor()

        plan = WorkflowPlan(
            id="empty-plan",
            source_vibe=Vibe(description="Empty plan test"),
            root_nodes=[]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 0
        assert len(result.execution_groups) == 0
        assert result.parallel_efficiency == 0.0

    async def test_single_node_parallel_execution(self):
        """Test parallel execution with single node."""
        executor = WorkflowExecutor()

        node = WorkflowNode(id="single", type=WorkflowNodeType.GENERATE, description="Single task")

        plan = WorkflowPlan(
            id="single-plan",
            source_vibe=Vibe(description="Single node test"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 1
        assert len(result.execution_groups) == 1
        assert len(result.execution_groups[0]) == 1
        assert result.execution_groups[0][0] == "single"