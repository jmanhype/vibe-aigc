"""Test workflow resume from checkpoint functionality."""

import pytest
import tempfile
import os
import shutil
import asyncio
from unittest.mock import patch, AsyncMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus, WorkflowExecutor
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager

@pytest.mark.asyncio
class TestWorkflowResume:
    """Test workflow checkpoint and resume functionality."""

    def setup_method(self):
        """Set up test environment with temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_executor_resume_from_checkpoint(self):
        """Test that WorkflowExecutor can resume from checkpoint correctly."""

        # Create a workflow with multiple nodes
        node1 = WorkflowNode(id="step-1", type=WorkflowNodeType.ANALYZE, description="First step")
        node2 = WorkflowNode(id="step-2", type=WorkflowNodeType.GENERATE,
                            description="Second step", dependencies=["step-1"])
        node3 = WorkflowNode(id="step-3", type=WorkflowNodeType.VALIDATE,
                            description="Third step", dependencies=["step-2"])

        plan = WorkflowPlan(
            id="resume-test-plan",
            source_vibe=Vibe(description="Resume test workflow"),
            root_nodes=[node1, node2, node3]
        )

        # Create a partial execution result (step-1 completed)
        partial_result = ExecutionResult("resume-test-plan")
        partial_result.add_node_result(NodeResult("step-1", ExecutionStatus.COMPLETED,
                                                 result={"analysis": "done"}, duration=0.5))

        # Create checkpoint with partial execution
        checkpoint = WorkflowCheckpoint(plan, partial_result)

        # Resume execution from checkpoint
        executor = WorkflowExecutor()
        result = await executor.execute_plan(plan, checkpoint)

        # Verify that all nodes completed
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3
        assert result.node_results["step-1"].status == ExecutionStatus.COMPLETED
        assert result.node_results["step-2"].status == ExecutionStatus.COMPLETED
        assert result.node_results["step-3"].status == ExecutionStatus.COMPLETED

        # Verify that step-1 was not re-executed (duration should remain the same)
        assert result.node_results["step-1"].duration == 0.5

    async def test_executor_resume_identifies_remaining_nodes(self):
        """Test that resume correctly identifies which nodes still need execution."""

        # Create workflow with parallel branches
        node1 = WorkflowNode(id="root", type=WorkflowNodeType.ANALYZE, description="Root analysis")
        node2a = WorkflowNode(id="branch-a", type=WorkflowNodeType.GENERATE,
                             description="Branch A", dependencies=["root"])
        node2b = WorkflowNode(id="branch-b", type=WorkflowNodeType.GENERATE,
                             description="Branch B", dependencies=["root"])
        node3 = WorkflowNode(id="merge", type=WorkflowNodeType.VALIDATE,
                            description="Merge results", dependencies=["branch-a", "branch-b"])

        plan = WorkflowPlan(
            id="parallel-resume-test",
            source_vibe=Vibe(description="Parallel resume test"),
            root_nodes=[node1, node2a, node2b, node3]
        )

        # Create partial execution (root and branch-a completed, branch-b failed)
        partial_result = ExecutionResult("parallel-resume-test")
        partial_result.add_node_result(NodeResult("root", ExecutionStatus.COMPLETED,
                                                 result={"analysis": "complete"}, duration=0.3))
        partial_result.add_node_result(NodeResult("branch-a", ExecutionStatus.COMPLETED,
                                                 result={"content": "generated"}, duration=0.4))
        partial_result.add_node_result(NodeResult("branch-b", ExecutionStatus.FAILED,
                                                 error="simulated failure", duration=0.2))

        checkpoint = WorkflowCheckpoint(plan, partial_result)

        # Resume execution
        executor = WorkflowExecutor()
        result = await executor.execute_plan(plan, checkpoint)

        # Verify final state
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 4

        # Verify that completed nodes were preserved
        assert result.node_results["root"].status == ExecutionStatus.COMPLETED
        assert result.node_results["branch-a"].status == ExecutionStatus.COMPLETED
        assert result.node_results["root"].duration == 0.3  # Original duration preserved
        assert result.node_results["branch-a"].duration == 0.4  # Original duration preserved

        # Verify that failed/missing nodes were executed
        assert result.node_results["branch-b"].status == ExecutionStatus.COMPLETED  # Re-executed
        assert result.node_results["merge"].status == ExecutionStatus.COMPLETED  # Now possible

    async def test_resume_with_progress_callbacks(self):
        """Test that progress callbacks work correctly during resume."""

        progress_events = []

        def capture_progress(event):
            progress_events.append(event)

        # Create workflow
        node1 = WorkflowNode(id="task-1", type=WorkflowNodeType.ANALYZE, description="First task")
        node2 = WorkflowNode(id="task-2", type=WorkflowNodeType.GENERATE, description="Second task")

        plan = WorkflowPlan(
            id="progress-resume-test",
            source_vibe=Vibe(description="Progress callback resume test"),
            root_nodes=[node1, node2]
        )

        # Create checkpoint with one completed task
        partial_result = ExecutionResult("progress-resume-test")
        partial_result.add_node_result(NodeResult("task-1", ExecutionStatus.COMPLETED,
                                                 result={"analysis": "done"}, duration=0.2))

        checkpoint = WorkflowCheckpoint(plan, partial_result)

        # Resume with progress callbacks
        executor = WorkflowExecutor(progress_callback=capture_progress)
        result = await executor.execute_plan(plan, checkpoint)

        # Verify execution completed
        assert result.status == ExecutionStatus.COMPLETED

        # Verify progress events were emitted
        assert len(progress_events) > 0

        # Should have workflow started, node events for task-2 (not task-1), workflow completed
        event_types = [event.event_type for event in progress_events]
        assert "workflow_started" in event_types
        assert "workflow_completed" in event_types

        # Check that resume was indicated in workflow started message
        workflow_started_events = [e for e in progress_events if e.event_type == "workflow_started"]
        assert len(workflow_started_events) > 0
        workflow_started_event = workflow_started_events[0]
        assert "resuming" in workflow_started_event.message.lower() or "resume" in workflow_started_event.message.lower()

    async def test_full_checkpoint_save_and_resume_cycle(self):
        """Test complete cycle: execute -> save checkpoint -> resume."""

        temp_dir = tempfile.mkdtemp()
        try:
            persistence_manager = WorkflowPersistenceManager(temp_dir)

            # Create multi-step workflow
            step1 = WorkflowNode(id="prepare", type=WorkflowNodeType.ANALYZE, description="Prepare data")
            step2 = WorkflowNode(id="process", type=WorkflowNodeType.GENERATE,
                               description="Process data", dependencies=["prepare"])
            step3 = WorkflowNode(id="validate", type=WorkflowNodeType.VALIDATE,
                               description="Validate results", dependencies=["process"])

            plan = WorkflowPlan(
                id="full-cycle-test",
                source_vibe=Vibe(description="Full checkpoint cycle test"),
                root_nodes=[step1, step2, step3]
            )

            # Simulate partial execution by creating checkpoint manually
            partial_result = ExecutionResult("full-cycle-test")
            partial_result.add_node_result(NodeResult("prepare", ExecutionStatus.COMPLETED,
                                                     result={"prepared": True}, duration=0.25))
            partial_result.add_node_result(NodeResult("process", ExecutionStatus.COMPLETED,
                                                     result={"processed_data": "content"}, duration=0.35))

            # Save checkpoint
            checkpoint = WorkflowCheckpoint(plan, partial_result)
            checkpoint_path = persistence_manager.save_checkpoint(checkpoint)
            assert os.path.exists(checkpoint_path)

            # Load checkpoint and resume
            loaded_checkpoint = persistence_manager.load_checkpoint(checkpoint.checkpoint_id)

            # Resume execution
            executor = WorkflowExecutor()
            result = await executor.execute_plan(plan, loaded_checkpoint)

            # Verify complete execution
            assert result.status == ExecutionStatus.COMPLETED
            assert len(result.node_results) == 3

            # Verify preserved results
            assert result.node_results["prepare"].duration == 0.25
            assert result.node_results["process"].duration == 0.35
            assert result.node_results["validate"].status == ExecutionStatus.COMPLETED

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
