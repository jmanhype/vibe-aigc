"""Tests for progress callback system."""

import pytest
import asyncio
from typing import List

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import (
    WorkflowExecutor, ExecutionStatus, ProgressEvent, ProgressEventType
)


class TestProgressCallbacks:
    """Test progress callback functionality."""

    @pytest.mark.asyncio
    async def test_basic_progress_events(self):
        """Test that basic progress events are emitted."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        node = WorkflowNode(
            id="progress-test",
            type=WorkflowNodeType.GENERATE,
            description="Progress test node"
        )
        plan = WorkflowPlan(
            id="progress-plan",
            source_vibe=Vibe(description="Test progress callbacks"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED

        # Verify progress events were emitted
        assert len(progress_events) >= 4  # At least: workflow_started, node_started, node_completed, workflow_completed

        event_types = [event.event_type for event in progress_events]

        # Check required event types
        assert ProgressEventType.WORKFLOW_STARTED in event_types
        assert ProgressEventType.NODE_STARTED in event_types
        assert ProgressEventType.NODE_COMPLETED in event_types
        assert ProgressEventType.WORKFLOW_COMPLETED in event_types

        # Check event order (workflow started should be first, completed should be last)
        assert progress_events[0].event_type == ProgressEventType.WORKFLOW_STARTED
        assert progress_events[-1].event_type == ProgressEventType.WORKFLOW_COMPLETED

        # Check final event shows 100% progress
        final_event = progress_events[-1]
        assert final_event.progress_percent == 100.0
        assert "completed" in final_event.message.lower()

    @pytest.mark.asyncio
    async def test_node_level_progress_events(self):
        """Test detailed node-level progress tracking."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        # Create multiple nodes
        node1 = WorkflowNode(id="node-1", type=WorkflowNodeType.ANALYZE, description="Analyze input")
        node2 = WorkflowNode(id="node-2", type=WorkflowNodeType.GENERATE, description="Generate content")

        plan = WorkflowPlan(
            id="multi-node-plan",
            source_vibe=Vibe(description="Multi-node progress test"),
            root_nodes=[node1, node2]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED

        # Find node-specific events
        node1_events = [e for e in progress_events if e.node_id == "node-1"]
        node2_events = [e for e in progress_events if e.node_id == "node-2"]

        # Each node should have start and completion events
        assert len(node1_events) >= 2  # At least started and completed
        assert len(node2_events) >= 2

        # Check node event types
        node1_types = [e.event_type for e in node1_events]
        assert ProgressEventType.NODE_STARTED in node1_types
        assert ProgressEventType.NODE_COMPLETED in node1_types

        # Verify messages contain node descriptions
        node1_started = next(e for e in node1_events if e.event_type == ProgressEventType.NODE_STARTED)
        assert "Analyze input" in node1_started.message

    @pytest.mark.asyncio
    async def test_parallel_group_progress_events(self):
        """Test progress events for parallel execution groups."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        # Create independent nodes for parallel execution
        node1 = WorkflowNode(id="parallel-1", type=WorkflowNodeType.GENERATE, description="Parallel task 1")
        node2 = WorkflowNode(id="parallel-2", type=WorkflowNodeType.GENERATE, description="Parallel task 2")
        node3 = WorkflowNode(id="parallel-3", type=WorkflowNodeType.GENERATE, description="Parallel task 3")

        plan = WorkflowPlan(
            id="parallel-progress-plan",
            source_vibe=Vibe(description="Parallel progress test"),
            root_nodes=[node1, node2, node3]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED

        # Find group events
        group_events = [e for e in progress_events if e.event_type in [
            ProgressEventType.GROUP_STARTED, ProgressEventType.GROUP_COMPLETED
        ]]

        # Should have group start and completion events
        assert len(group_events) >= 2

        group_started_events = [e for e in group_events if e.event_type == ProgressEventType.GROUP_STARTED]
        group_completed_events = [e for e in group_events if e.event_type == ProgressEventType.GROUP_COMPLETED]

        assert len(group_started_events) > 0
        assert len(group_completed_events) > 0

        # Group events should have metadata
        group_started = group_started_events[0]
        assert "group_size" in group_started.metadata
        assert "group_nodes" in group_started.metadata
        assert group_started.metadata["group_size"] == 3  # All three nodes in one parallel group

    @pytest.mark.asyncio
    async def test_dependency_based_groups_progress(self):
        """Test progress tracking with dependency-based execution groups."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        # Create dependency chain: step1 -> step2, plus independent step3
        step1 = WorkflowNode(id="step-1", type=WorkflowNodeType.ANALYZE, description="First step")
        step2 = WorkflowNode(id="step-2", type=WorkflowNodeType.GENERATE,
                           description="Second step", dependencies=["step-1"])
        step3 = WorkflowNode(id="step-3", type=WorkflowNodeType.VALIDATE, description="Independent step")

        plan = WorkflowPlan(
            id="dependency-progress-plan",
            source_vibe=Vibe(description="Dependency progress test"),
            root_nodes=[step1, step2, step3]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED

        # Should have multiple groups due to dependencies
        group_started_events = [e for e in progress_events if e.event_type == ProgressEventType.GROUP_STARTED]

        # Should have at least 2 groups: [step1, step3] and [step2]
        assert len(group_started_events) >= 2

        # Check progress percentages increase
        group_completed_events = [e for e in progress_events if e.event_type == ProgressEventType.GROUP_COMPLETED]
        if len(group_completed_events) > 1:
            # Progress should increase between groups
            assert group_completed_events[0].progress_percent < group_completed_events[-1].progress_percent

    @pytest.mark.asyncio
    async def test_hierarchical_progress_tracking(self):
        """Test progress tracking with hierarchical workflows (parent-child)."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        # Create hierarchical structure
        child1 = WorkflowNode(id="child-1", type=WorkflowNodeType.ANALYZE, description="Child 1")
        child2 = WorkflowNode(id="child-2", type=WorkflowNodeType.GENERATE, description="Child 2")

        parent = WorkflowNode(
            id="parent",
            type=WorkflowNodeType.COMPOSITE,
            description="Parent with children",
            children=[child1, child2]
        )

        plan = WorkflowPlan(
            id="hierarchical-progress-plan",
            source_vibe=Vibe(description="Hierarchical progress test"),
            root_nodes=[parent]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED

        # Should have events for parent and children
        node_ids_in_events = set(e.node_id for e in progress_events if e.node_id)
        assert "parent" in node_ids_in_events
        assert "child-1" in node_ids_in_events
        assert "child-2" in node_ids_in_events

        # Total nodes should include parent + children
        total_nodes = 3  # parent + 2 children
        workflow_events = [e for e in progress_events if e.event_type == ProgressEventType.WORKFLOW_COMPLETED]
        assert len(workflow_events) == 1
        assert workflow_events[0].progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_failure_progress_events(self):
        """Test progress events when nodes fail."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        # Mock handler to simulate failure
        original_handler = executor._execute_generate

        async def failing_handler(node):
            if node.id == "fail-node":
                raise RuntimeError("Simulated failure")
            return await original_handler(node)

        executor.node_handlers[WorkflowNodeType.GENERATE] = failing_handler

        failing_node = WorkflowNode(
            id="fail-node",
            type=WorkflowNodeType.GENERATE,
            description="This node will fail"
        )

        plan = WorkflowPlan(
            id="failure-progress-plan",
            source_vibe=Vibe(description="Test failure progress events"),
            root_nodes=[failing_node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.FAILED

        # Should have failure-specific events
        event_types = [event.event_type for event in progress_events]
        assert ProgressEventType.NODE_FAILED in event_types

        # Find the failure event
        failure_events = [e for e in progress_events if e.event_type == ProgressEventType.NODE_FAILED]
        assert len(failure_events) == 1

        failure_event = failure_events[0]
        assert failure_event.node_id == "fail-node"
        assert "Simulated failure" in failure_event.message
        assert "error" in failure_event.metadata
        assert failure_event.metadata["error"] == "Simulated failure"

    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test that callback failures don't interrupt execution."""

        def failing_callback(event: ProgressEvent):
            raise RuntimeError("Callback failed!")

        executor = WorkflowExecutor(progress_callback=failing_callback)

        node = WorkflowNode(id="robust-test", type=WorkflowNodeType.GENERATE, description="Test robustness")
        plan = WorkflowPlan(
            id="robust-plan",
            source_vibe=Vibe(description="Test callback robustness"),
            root_nodes=[node]
        )

        # Execution should succeed despite callback failures
        result = await executor.execute_plan(plan)
        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_callback_execution(self):
        """Test that execution works normally when no callback is provided."""

        # No callback provided
        executor = WorkflowExecutor()

        node = WorkflowNode(id="no-callback", type=WorkflowNodeType.GENERATE, description="No callback test")
        plan = WorkflowPlan(
            id="no-callback-plan",
            source_vibe=Vibe(description="Test without callbacks"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        # Should work normally
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 1
        assert result.node_results["no-callback"].status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_progress_event_timestamps(self):
        """Test that progress events have proper timestamps."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        node = WorkflowNode(id="timestamp-test", type=WorkflowNodeType.GENERATE, description="Timestamp test")
        plan = WorkflowPlan(
            id="timestamp-plan",
            source_vibe=Vibe(description="Test event timestamps"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)
        assert result.status == ExecutionStatus.COMPLETED

        # All events should have timestamps
        for event in progress_events:
            assert event.timestamp is not None
            assert len(event.timestamp) > 0

        # Timestamps should be in order (approximately)
        timestamps = [event.timestamp for event in progress_events]
        # Check first vs last (should be chronologically ordered)
        assert timestamps[0] <= timestamps[-1]

    @pytest.mark.asyncio
    async def test_progress_metadata_content(self):
        """Test that progress events contain useful metadata."""

        progress_events: List[ProgressEvent] = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        node = WorkflowNode(id="metadata-test", type=WorkflowNodeType.ANALYZE, description="Metadata test")
        plan = WorkflowPlan(
            id="metadata-plan",
            source_vibe=Vibe(description="Test metadata content"),
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)
        assert result.status == ExecutionStatus.COMPLETED

        # Check node completed event has useful metadata
        completed_events = [e for e in progress_events if e.event_type == ProgressEventType.NODE_COMPLETED]
        assert len(completed_events) == 1

        completed_event = completed_events[0]
        assert "duration" in completed_event.metadata
        assert "result_preview" in completed_event.metadata
        assert completed_event.metadata["duration"] > 0  # Should have some execution time