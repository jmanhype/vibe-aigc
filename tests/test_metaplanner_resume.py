"""Test MetaPlanner resume integration functionality."""

import pytest
import tempfile
import os
import shutil
from unittest.mock import patch, AsyncMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager

@pytest.mark.asyncio
class TestMetaPlannerResume:
    """Test MetaPlanner resume integration functionality."""

    def setup_method(self):
        """Set up test environment with temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('vibe_aigc.planner.LLMClient')
    async def test_execute_with_resume_from_checkpoint(self, mock_llm_client):
        """Test MetaPlanner.execute_with_resume() with existing checkpoint."""

        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "resume-metaplanner-test",
            "root_nodes": [
                {"id": "step-1", "type": "analyze", "description": "First step",
                 "parameters": {}, "dependencies": [], "children": []},
                {"id": "step-2", "type": "generate", "description": "Second step",
                 "parameters": {}, "dependencies": ["step-1"], "children": []},
                {"id": "step-3", "type": "validate", "description": "Third step",
                 "parameters": {}, "dependencies": ["step-2"], "children": []}
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create a pre-existing checkpoint with partial execution
        step1 = WorkflowNode(id="step-1", type=WorkflowNodeType.ANALYZE, description="First step")
        step2 = WorkflowNode(id="step-2", type=WorkflowNodeType.GENERATE,
                            description="Second step", dependencies=["step-1"])
        step3 = WorkflowNode(id="step-3", type=WorkflowNodeType.VALIDATE,
                            description="Third step", dependencies=["step-2"])

        plan = WorkflowPlan(
            id="resume-metaplanner-test",
            source_vibe=Vibe(description="MetaPlanner resume test"),
            root_nodes=[step1, step2, step3]
        )

        # Create partial execution result
        partial_result = ExecutionResult("resume-metaplanner-test")
        partial_result.add_node_result(NodeResult("step-1", ExecutionStatus.COMPLETED,
                                                 result={"analysis": "done"}, duration=0.5))

        # Save checkpoint
        checkpoint = WorkflowCheckpoint(plan, partial_result)
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)
        persistence_manager.save_checkpoint(checkpoint)

        # Create MetaPlanner and resume execution
        planner = MetaPlanner(checkpoint_dir=self.temp_dir)
        vibe = Vibe(description="MetaPlanner resume test")

        result = await planner.execute_with_resume(vibe, checkpoint.checkpoint_id)

        # Verify execution completed successfully
        assert result["status"] == "completed"
        assert result["plan_id"] == "resume-metaplanner-test"

        # Verify resume metadata
        assert "resumed_from" in result
        assert result["resumed_from"] == checkpoint.checkpoint_id

        # Verify original step-1 result was preserved
        assert "step-1" in result["node_results"]
        assert result["node_results"]["step-1"]["duration"] == 0.5  # Original duration preserved
        assert result["node_results"]["step-1"]["result"]["analysis"] == "done"  # Original result preserved

    async def test_list_checkpoints_functionality(self):
        """Test MetaPlanner.list_checkpoints() functionality."""

        # Create multiple checkpoints
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)

        for i in range(3):
            node = WorkflowNode(id=f"task-{i+1}", type=WorkflowNodeType.ANALYZE,
                               description=f"Test task {i+1}")
            plan = WorkflowPlan(
                id=f"test-plan-{i+1}",
                source_vibe=Vibe(description=f"Test plan {i+1}"),
                root_nodes=[node]
            )
            result = ExecutionResult(f"test-plan-{i+1}")
            result.add_node_result(NodeResult(f"task-{i+1}", ExecutionStatus.COMPLETED,
                                             result={"completed": True}, duration=0.1))

            checkpoint = WorkflowCheckpoint(plan, result)
            persistence_manager.save_checkpoint(checkpoint)

        # Create MetaPlanner and list checkpoints
        planner = MetaPlanner(checkpoint_dir=self.temp_dir)

        # Test listing all checkpoints
        all_checkpoints = planner.list_checkpoints()
        assert len(all_checkpoints) == 3

        # Verify checkpoint metadata
        for checkpoint_info in all_checkpoints:
            assert "checkpoint_id" in checkpoint_info
            assert "plan_id" in checkpoint_info
            assert "created_at" in checkpoint_info
            assert checkpoint_info["plan_id"].startswith("test-plan-")

        # Test filtering by plan ID
        filtered_checkpoints = planner.list_checkpoints(plan_id="test-plan-2")
        assert len(filtered_checkpoints) == 1
        assert filtered_checkpoints[0]["plan_id"] == "test-plan-2"

        # Test filtering with non-existent plan ID
        empty_checkpoints = planner.list_checkpoints(plan_id="non-existent-plan")
        assert len(empty_checkpoints) == 0

    async def test_delete_checkpoint_functionality(self):
        """Test MetaPlanner.delete_checkpoint() functionality."""

        # Create a checkpoint
        node = WorkflowNode(id="delete-task", type=WorkflowNodeType.ANALYZE, description="Delete test task")
        plan = WorkflowPlan(
            id="delete-test-plan",
            source_vibe=Vibe(description="Delete test plan"),
            root_nodes=[node]
        )
        result = ExecutionResult("delete-test-plan")
        result.add_node_result(NodeResult("delete-task", ExecutionStatus.COMPLETED,
                                         result={"completed": True}, duration=0.1))

        checkpoint = WorkflowCheckpoint(plan, result)
        persistence_manager = WorkflowPersistenceManager(self.temp_dir)
        persistence_manager.save_checkpoint(checkpoint)

        # Create MetaPlanner
        planner = MetaPlanner(checkpoint_dir=self.temp_dir)

        # Verify checkpoint exists
        checkpoints = planner.list_checkpoints()
        assert len(checkpoints) == 1
        checkpoint_id = checkpoints[0]["checkpoint_id"]

        # Delete checkpoint
        deleted = planner.delete_checkpoint(checkpoint_id)
        assert deleted == True

        # Verify checkpoint is gone
        checkpoints_after = planner.list_checkpoints()
        assert len(checkpoints_after) == 0

        # Test deleting non-existent checkpoint
        deleted_nonexistent = planner.delete_checkpoint("non-existent-id")
        assert deleted_nonexistent == False

    async def test_resume_error_handling_checkpoint_not_found(self):
        """Test error handling when checkpoint is not found."""

        planner = MetaPlanner(checkpoint_dir=self.temp_dir)
        vibe = Vibe(description="Error handling test")

        # Try to resume from non-existent checkpoint
        with pytest.raises(RuntimeError) as exc_info:
            await planner.execute_with_resume(vibe, "non-existent-checkpoint-id")

        assert "Checkpoint not found: non-existent-checkpoint-id" in str(exc_info.value)
        assert "Use list_checkpoints() to see available checkpoints" in str(exc_info.value)
