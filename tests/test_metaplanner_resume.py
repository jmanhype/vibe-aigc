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
    async def test_list_checkpoints_functionality(self, mock_llm_client):
        """Test MetaPlanner.list_checkpoints() functionality."""

        # Mock LLM client
        mock_client = AsyncMock()
        mock_llm_client.return_value = mock_client

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

    @patch('vibe_aigc.planner.LLMClient')
    async def test_delete_checkpoint_functionality(self, mock_llm_client):
        """Test MetaPlanner.delete_checkpoint() functionality."""

        # Mock LLM client
        mock_client = AsyncMock()
        mock_llm_client.return_value = mock_client

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

    @patch('vibe_aigc.planner.LLMClient')
    async def test_resume_error_handling_checkpoint_not_found(self, mock_llm_client):
        """Test error handling when checkpoint is not found."""

        # Mock LLM client
        mock_client = AsyncMock()
        mock_llm_client.return_value = mock_client

        planner = MetaPlanner(checkpoint_dir=self.temp_dir)
        vibe = Vibe(description="Error handling test")

        # Try to resume from non-existent checkpoint
        with pytest.raises(RuntimeError) as exc_info:
            await planner.execute_with_resume(vibe, "non-existent-checkpoint-id")

        assert "Checkpoint not found: non-existent-checkpoint-id" in str(exc_info.value)
        assert "Use list_checkpoints() to see available checkpoints" in str(exc_info.value)

    @patch('vibe_aigc.planner.LLMClient')
    async def test_execute_with_resume_normal_execution(self, mock_llm_client):
        """Test MetaPlanner.execute_with_resume() without checkpoint (normal execution)."""

        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "normal-execution-test",
            "root_nodes": [
                {"id": "normal-task", "type": "analyze", "description": "Normal task",
                 "parameters": {}, "dependencies": [], "children": []}
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create MetaPlanner
        planner = MetaPlanner(checkpoint_dir=self.temp_dir)
        vibe = Vibe(description="Normal execution through resume interface")

        # Execute without checkpoint ID (should be normal execution)
        result = await planner.execute_with_resume(vibe)

        # Verify execution completed successfully
        assert result["status"] == "completed"
        assert result["plan_id"] == "normal-execution-test"

        # Verify no resume metadata for normal execution
        assert result["persistence_info"]["resumed_from"] is None

        # Verify task completed
        assert "normal-task" in result["node_results"]
        assert result["node_results"]["normal-task"]["status"] == "completed"