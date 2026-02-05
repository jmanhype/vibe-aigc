"""Tests for MetaPlanner resume integration (US-015)."""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, ExecutionStatus, NodeResult
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.llm import LLMConfig
from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager


def create_test_planner(checkpoint_dir: str, checkpoint_interval: int = None) -> MetaPlanner:
    """Create a MetaPlanner with mocked LLM config."""
    # Use a fake API key to avoid OpenAI client initialization error
    config = LLMConfig(api_key="test-fake-key-for-testing")
    return MetaPlanner(
        llm_config=config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval
    )


def create_test_plan(vibe: Vibe, num_nodes: int = 3) -> WorkflowPlan:
    """Create a test workflow plan."""
    nodes = [
        WorkflowNode(
            id=f"node-{i}",
            type=WorkflowNodeType.GENERATE,
            description=f"Test node {i}"
        )
        for i in range(num_nodes)
    ]
    return WorkflowPlan(id="test-plan", source_vibe=vibe, root_nodes=nodes)


def create_partial_checkpoint(plan: WorkflowPlan, completed_ids: list, tmp_path) -> str:
    """Create a checkpoint with partial execution."""
    result = ExecutionResult(plan.id)

    for node in plan.root_nodes:
        if node.id in completed_ids:
            result.add_node_result(NodeResult(
                node_id=node.id,
                status=ExecutionStatus.COMPLETED,
                result={"mock": "result"},
                duration=0.1
            ))

    checkpoint = WorkflowCheckpoint(plan, result)
    manager = WorkflowPersistenceManager(str(tmp_path))
    manager.save_checkpoint(checkpoint)

    return checkpoint.checkpoint_id


@pytest.mark.asyncio
class TestMetaPlannerResumeIntegration:
    """Test MetaPlanner checkpoint/resume integration."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_execute_with_resume_fresh_start(self, mock_openai, tmp_path):
        """Test execute_with_resume without checkpoint (fresh execution)."""
        # Mock LLM response
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "fresh-plan",
            "root_nodes": [
                {"id": "task-1", "type": "generate", "description": "First task"}
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        planner = create_test_planner(str(tmp_path))
        vibe = Vibe(description="Test fresh execution")

        result = await planner.execute_with_resume(vibe)

        assert result["status"] == "completed"
        assert result["persistence_info"]["resumed_from"] is None

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_execute_with_resume_from_checkpoint(self, mock_openai, tmp_path):
        """Test execute_with_resume resuming from checkpoint."""
        # Create a plan and partial checkpoint
        vibe = Vibe(description="Test resume execution")
        plan = create_test_plan(vibe, num_nodes=4)
        checkpoint_id = create_partial_checkpoint(plan, ["node-0", "node-1"], tmp_path)

        planner = create_test_planner(str(tmp_path))

        result = await planner.execute_with_resume(vibe, checkpoint_id=checkpoint_id)

        assert result["status"] == "completed"
        assert result["persistence_info"]["resumed_from"] == checkpoint_id
        # All 4 nodes should be in results
        assert len(result["node_results"]) == 4

    async def test_execute_with_resume_invalid_checkpoint(self, tmp_path):
        """Test execute_with_resume with non-existent checkpoint."""
        planner = create_test_planner(str(tmp_path))
        vibe = Vibe(description="Test invalid checkpoint")

        with pytest.raises(RuntimeError, match="Checkpoint not found"):
            await planner.execute_with_resume(vibe, checkpoint_id="nonexistent-id")

    def test_list_checkpoints_empty(self, tmp_path):
        """Test listing checkpoints when none exist."""
        planner = create_test_planner(str(tmp_path))

        checkpoints = planner.list_checkpoints()

        assert checkpoints == []

    def test_list_checkpoints_with_data(self, tmp_path):
        """Test listing checkpoints with existing data."""
        vibe = Vibe(description="Test list")
        plan = create_test_plan(vibe, num_nodes=2)

        # Create some checkpoints
        manager = WorkflowPersistenceManager(str(tmp_path))

        result1 = ExecutionResult(plan.id)
        result1.add_node_result(NodeResult("node-0", ExecutionStatus.COMPLETED))
        checkpoint1 = WorkflowCheckpoint(plan, result1)
        manager.save_checkpoint(checkpoint1)

        result2 = ExecutionResult(plan.id)
        result2.add_node_result(NodeResult("node-0", ExecutionStatus.COMPLETED))
        result2.add_node_result(NodeResult("node-1", ExecutionStatus.COMPLETED))
        checkpoint2 = WorkflowCheckpoint(plan, result2)
        manager.save_checkpoint(checkpoint2)

        # List via MetaPlanner
        planner = create_test_planner(str(tmp_path))
        checkpoints = planner.list_checkpoints()

        assert len(checkpoints) == 2

    def test_list_checkpoints_filter_by_plan_id(self, tmp_path):
        """Test listing checkpoints filtered by plan ID."""
        vibe1 = Vibe(description="Plan 1")
        plan1 = WorkflowPlan(id="plan-one", source_vibe=vibe1, root_nodes=[])

        vibe2 = Vibe(description="Plan 2")
        plan2 = WorkflowPlan(id="plan-two", source_vibe=vibe2, root_nodes=[])

        manager = WorkflowPersistenceManager(str(tmp_path))

        # Create checkpoints for both plans
        cp1 = WorkflowCheckpoint(plan1, ExecutionResult(plan1.id))
        cp2 = WorkflowCheckpoint(plan2, ExecutionResult(plan2.id))
        manager.save_checkpoint(cp1)
        manager.save_checkpoint(cp2)

        planner = create_test_planner(str(tmp_path))

        # Filter by plan ID
        plan_one_checkpoints = planner.list_checkpoints(plan_id="plan-one")

        assert len(plan_one_checkpoints) == 1
        assert plan_one_checkpoints[0]["plan_id"] == "plan-one"

    def test_delete_checkpoint(self, tmp_path):
        """Test deleting a checkpoint."""
        vibe = Vibe(description="Delete test")
        plan = create_test_plan(vibe)

        manager = WorkflowPersistenceManager(str(tmp_path))
        checkpoint = WorkflowCheckpoint(plan, ExecutionResult(plan.id))
        manager.save_checkpoint(checkpoint)

        planner = create_test_planner(str(tmp_path))

        # Verify checkpoint exists
        assert len(planner.list_checkpoints()) == 1

        # Delete it
        result = planner.delete_checkpoint(checkpoint.checkpoint_id)

        assert result is True
        assert len(planner.list_checkpoints()) == 0

    def test_delete_checkpoint_not_found(self, tmp_path):
        """Test deleting non-existent checkpoint returns False."""
        planner = create_test_planner(str(tmp_path))

        result = planner.delete_checkpoint("nonexistent-checkpoint")

        assert result is False

    def test_get_checkpoint(self, tmp_path):
        """Test loading a specific checkpoint."""
        vibe = Vibe(description="Get test")
        plan = create_test_plan(vibe)

        manager = WorkflowPersistenceManager(str(tmp_path))
        original = WorkflowCheckpoint(plan, ExecutionResult(plan.id))
        manager.save_checkpoint(original)

        planner = create_test_planner(str(tmp_path))
        loaded = planner.get_checkpoint(original.checkpoint_id)

        assert loaded.checkpoint_id == original.checkpoint_id
        assert loaded.plan.id == plan.id

    def test_get_checkpoint_not_found(self, tmp_path):
        """Test loading non-existent checkpoint raises FileNotFoundError."""
        planner = create_test_planner(str(tmp_path))

        with pytest.raises(FileNotFoundError):
            planner.get_checkpoint("nonexistent-id")

    def test_create_checkpoint_manual(self, tmp_path):
        """Test manually creating a checkpoint."""
        vibe = Vibe(description="Manual checkpoint test")
        plan = create_test_plan(vibe)
        result = ExecutionResult(plan.id)
        result.add_node_result(NodeResult("node-0", ExecutionStatus.COMPLETED))

        planner = create_test_planner(str(tmp_path))
        checkpoint_id = planner.create_checkpoint(plan, result)

        # Verify it was created
        assert checkpoint_id is not None
        loaded = planner.get_checkpoint(checkpoint_id)
        assert loaded.plan.id == plan.id

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_persistence_info_in_result(self, mock_openai, tmp_path):
        """Test that execute_with_resume includes persistence info."""
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "persist-plan",
            "root_nodes": [
                {"id": "task-1", "type": "generate", "description": "Task"}
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        planner = create_test_planner(str(tmp_path), checkpoint_interval=1)
        vibe = Vibe(description="Persistence info test")

        result = await planner.execute_with_resume(vibe)

        assert "persistence_info" in result
        assert result["persistence_info"]["checkpoint_dir"] == str(tmp_path)
        assert result["persistence_info"]["checkpoint_enabled"] is True

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_backward_compat_execute_method(self, mock_openai, tmp_path):
        """Test that original execute() method still works."""
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "compat-plan",
            "root_nodes": [
                {"id": "task-1", "type": "generate", "description": "Task"}
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        planner = create_test_planner(str(tmp_path))
        vibe = Vibe(description="Backward compat test")

        # Original execute should still work
        result = await planner.execute(vibe)

        assert result["status"] == "completed"


@pytest.mark.asyncio
class TestResumeFullCycle:
    """Integration tests for full checkpoint/resume cycle via MetaPlanner."""

    async def test_full_checkpoint_resume_cycle(self, tmp_path):
        """Test complete cycle: execute partially, save, list, load, resume."""
        vibe = Vibe(description="Full cycle test")
        plan = create_test_plan(vibe, num_nodes=4)

        # Step 1: Create partial execution and checkpoint
        result = ExecutionResult(plan.id)
        result.add_node_result(NodeResult("node-0", ExecutionStatus.COMPLETED,
                                         result={"data": "first"}))
        result.add_node_result(NodeResult("node-1", ExecutionStatus.COMPLETED,
                                         result={"data": "second"}))

        planner = create_test_planner(str(tmp_path))
        checkpoint_id = planner.create_checkpoint(plan, result)

        # Step 2: List checkpoints
        checkpoints = planner.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["checkpoint_id"] == checkpoint_id

        # Step 3: Resume execution
        final_result = await planner.execute_with_resume(vibe, checkpoint_id=checkpoint_id)

        # Step 4: Verify
        assert final_result["status"] == "completed"
        assert final_result["persistence_info"]["resumed_from"] == checkpoint_id
        assert len(final_result["node_results"]) == 4

        # Previously completed nodes should retain their results
        assert final_result["node_results"]["node-0"]["status"] == "completed"
        assert final_result["node_results"]["node-1"]["status"] == "completed"
