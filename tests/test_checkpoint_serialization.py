"""Test workflow checkpoint serialization and persistence functionality."""

import pytest
import tempfile
import os
import shutil
import json

from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager
from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus


class TestWorkflowCheckpoint:
    """Test WorkflowCheckpoint serialization and deserialization."""

    def test_checkpoint_creation(self):
        """Test basic WorkflowCheckpoint creation."""

        # Create test workflow
        node = WorkflowNode(
            id="test-node",
            type=WorkflowNodeType.GENERATE,
            description="Test node"
        )

        vibe = Vibe(
            description="Test vibe",
            constraints=["test constraint"],
            metadata={"test": "value"}
        )

        plan = WorkflowPlan(
            id="test-plan",
            source_vibe=vibe,
            root_nodes=[node]
        )

        # Create execution result
        result = ExecutionResult("test-plan")
        result.status = ExecutionStatus.RUNNING
        result.add_node_result(NodeResult(
            "test-node",
            ExecutionStatus.COMPLETED,
            result={"output": "test result"},
            duration=1.5
        ))

        # Create checkpoint
        checkpoint = WorkflowCheckpoint(plan, result)

        # Verify checkpoint properties
        assert checkpoint.plan == plan
        assert checkpoint.execution_result == result
        assert checkpoint.schema_version == "1.0"
        assert checkpoint.checkpoint_id.startswith("test-plan_")
        assert checkpoint.created_at is not None

    def test_checkpoint_serialization_roundtrip(self):
        """Test that checkpoints can be serialized and deserialized correctly."""

        # Create complex test workflow
        child_node = WorkflowNode(
            id="child-node",
            type=WorkflowNodeType.VALIDATE,
            description="Child validation",
            parameters={"threshold": 0.8}
        )

        parent_node = WorkflowNode(
            id="parent-node",
            type=WorkflowNodeType.COMPOSITE,
            description="Parent composite",
            children=[child_node],
            dependencies=["dependency-1"],
            estimated_duration=300
        )

        vibe = Vibe(
            description="Complex test vibe",
            style="sophisticated",
            constraints=["high quality", "fast execution"],
            domain="testing",
            metadata={"version": "1.0", "priority": "high"}
        )

        plan = WorkflowPlan(
            id="complex-plan",
            source_vibe=vibe,
            root_nodes=[parent_node],
            estimated_total_duration=600,
            created_at="2026-02-05T10:00:00"
        )

        # Create detailed execution result
        result = ExecutionResult("complex-plan")
        result.status = ExecutionStatus.COMPLETED
        result.started_at = "2026-02-05T10:00:00"
        result.completed_at = "2026-02-05T10:05:00"
        result.total_duration = 300.5

        # Add node results
        result.add_node_result(NodeResult(
            "parent-node",
            ExecutionStatus.COMPLETED,
            result={"composite_output": "success"},
            duration=200.0
        ))

        result.add_node_result(NodeResult(
            "child-node",
            ExecutionStatus.COMPLETED,
            result={"validation_score": 0.95},
            duration=100.0
        ))

        # Add extended fields
        result.parallel_efficiency = 0.35
        result.execution_groups = [["parent-node"], ["child-node"]]
        result.feedback_data = {"parent-node": {"quality": 0.9}}
        result.replan_suggestions = [{"reason": "optimization", "suggestion": "parallelize"}]

        # Create and serialize checkpoint
        checkpoint = WorkflowCheckpoint(plan, result)
        checkpoint_data = checkpoint.to_dict()

        # Deserialize and verify
        restored_checkpoint = WorkflowCheckpoint.from_dict(checkpoint_data)

        # Verify plan restoration
        assert restored_checkpoint.plan.id == plan.id
        assert restored_checkpoint.plan.source_vibe.description == vibe.description
        assert restored_checkpoint.plan.source_vibe.style == vibe.style
        assert restored_checkpoint.plan.source_vibe.constraints == vibe.constraints
        assert restored_checkpoint.plan.source_vibe.domain == vibe.domain
        assert restored_checkpoint.plan.source_vibe.metadata == vibe.metadata

        # Verify node structure
        assert len(restored_checkpoint.plan.root_nodes) == 1
        restored_parent = restored_checkpoint.plan.root_nodes[0]
        assert restored_parent.id == "parent-node"
        assert restored_parent.type == WorkflowNodeType.COMPOSITE
        assert restored_parent.description == "Parent composite"
        assert restored_parent.dependencies == ["dependency-1"]
        assert restored_parent.estimated_duration == 300

        # Verify child node
        assert len(restored_parent.children) == 1
        restored_child = restored_parent.children[0]
        assert restored_child.id == "child-node"
        assert restored_child.type == WorkflowNodeType.VALIDATE
        assert restored_child.parameters == {"threshold": 0.8}

        # Verify execution result
        restored_result = restored_checkpoint.execution_result
        assert restored_result.plan_id == result.plan_id
        assert restored_result.status == result.status
        assert restored_result.started_at == result.started_at
        assert restored_result.completed_at == result.completed_at
        assert restored_result.total_duration == result.total_duration

        # Verify node results
        assert len(restored_result.node_results) == 2
        assert "parent-node" in restored_result.node_results
        assert "child-node" in restored_result.node_results

        parent_result = restored_result.node_results["parent-node"]
        assert parent_result.status == ExecutionStatus.COMPLETED
        assert parent_result.result == {"composite_output": "success"}
        assert parent_result.duration == 200.0

        # Verify extended fields
        assert restored_result.parallel_efficiency == 0.35
        assert restored_result.execution_groups == [["parent-node"], ["child-node"]]
        assert restored_result.feedback_data == {"parent-node": {"quality": 0.9}}
        assert restored_result.replan_suggestions == [{"reason": "optimization", "suggestion": "parallelize"}]

    def test_checkpoint_id_generation(self):
        """Test checkpoint ID generation is unique and follows expected format."""

        node = WorkflowNode(id="test", type=WorkflowNodeType.GENERATE, description="Test")
        plan = WorkflowPlan(id="test-plan", source_vibe=Vibe(description="Test"), root_nodes=[node])
        result = ExecutionResult("test-plan")

        # Create multiple checkpoints
        checkpoint1 = WorkflowCheckpoint(plan, result)
        checkpoint2 = WorkflowCheckpoint(plan, result)

        # IDs should be unique
        assert checkpoint1.checkpoint_id != checkpoint2.checkpoint_id

        # IDs should follow expected format: plan_id_timestamp_hash
        assert checkpoint1.checkpoint_id.startswith("test-plan_")
        assert len(checkpoint1.checkpoint_id.split("_")) >= 4  # plan, date, time, hash

    def test_schema_version_validation(self):
        """Test schema version validation during deserialization."""

        node = WorkflowNode(id="test", type=WorkflowNodeType.GENERATE, description="Test")
        plan = WorkflowPlan(id="test", source_vibe=Vibe(description="Test"), root_nodes=[node])
        result = ExecutionResult("test")

        checkpoint = WorkflowCheckpoint(plan, result)
        data = checkpoint.to_dict()

        # Modify schema version
        data["schema_version"] = "2.0"

        # Should raise ValueError for unsupported version
        with pytest.raises(ValueError, match="Unsupported checkpoint schema version"):
            WorkflowCheckpoint.from_dict(data)

    def test_serialization_handles_none_values(self):
        """Test serialization handles None values gracefully."""

        node = WorkflowNode(
            id="minimal-node",
            type=WorkflowNodeType.GENERATE,
            description="Minimal node"
        )

        vibe = Vibe(description="Minimal vibe")  # Only required field

        plan = WorkflowPlan(
            id="minimal-plan",
            source_vibe=vibe,
            root_nodes=[node]
        )

        result = ExecutionResult("minimal-plan")
        # Don't set optional fields

        checkpoint = WorkflowCheckpoint(plan, result)
        data = checkpoint.to_dict()

        # Should serialize without errors
        restored_checkpoint = WorkflowCheckpoint.from_dict(data)

        # Verify None values are handled
        assert restored_checkpoint.plan.estimated_total_duration is None
        assert restored_checkpoint.plan.created_at is None
        assert restored_checkpoint.plan.source_vibe.style is None
        assert restored_checkpoint.plan.source_vibe.domain is None


class TestWorkflowPersistenceManager:
    """Test WorkflowPersistenceManager functionality."""

    def setup_method(self):
        """Set up test environment with temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_manager = WorkflowPersistenceManager(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints through persistence manager."""

        # Create test checkpoint
        node = WorkflowNode(id="persist-node", type=WorkflowNodeType.ANALYZE, description="Persist test")
        plan = WorkflowPlan(
            id="persist-plan",
            source_vibe=Vibe(description="Test persistence"),
            root_nodes=[node]
        )
        result = ExecutionResult("persist-plan")
        result.add_node_result(NodeResult("persist-node", ExecutionStatus.COMPLETED, duration=2.0))

        checkpoint = WorkflowCheckpoint(plan, result)

        # Save checkpoint
        saved_path = self.persistence_manager.save_checkpoint(checkpoint)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(f"{checkpoint.checkpoint_id}.json")

        # Load checkpoint
        loaded_checkpoint = self.persistence_manager.load_checkpoint(checkpoint.checkpoint_id)

        assert loaded_checkpoint.plan.id == plan.id
        assert loaded_checkpoint.execution_result.plan_id == result.plan_id
        assert len(loaded_checkpoint.execution_result.node_results) == 1

    def test_checkpoint_listing(self):
        """Test listing available checkpoints with metadata."""

        # Create multiple checkpoints
        for i in range(3):
            node = WorkflowNode(id=f"node-{i}", type=WorkflowNodeType.GENERATE, description=f"Test node {i}")
            plan = WorkflowPlan(
                id=f"plan-{i}",
                source_vibe=Vibe(description=f"Test vibe {i}"),
                root_nodes=[node]
            )
            result = ExecutionResult(f"plan-{i}")
            result.status = ExecutionStatus.COMPLETED if i % 2 == 0 else ExecutionStatus.FAILED

            checkpoint = WorkflowCheckpoint(plan, result)
            self.persistence_manager.save_checkpoint(checkpoint)

        # List checkpoints
        checkpoints = self.persistence_manager.list_checkpoints()

        assert len(checkpoints) == 3
        for checkpoint_info in checkpoints:
            assert "checkpoint_id" in checkpoint_info
            assert "plan_id" in checkpoint_info
            assert "created_at" in checkpoint_info
            assert "status" in checkpoint_info
            assert "vibe_description" in checkpoint_info

            # Verify vibe description is truncated to 50 chars
            assert len(checkpoint_info["vibe_description"]) <= 50

        # Should be sorted by created_at descending (newest first)
        created_times = [cp["created_at"] for cp in checkpoints]
        assert created_times == sorted(created_times, reverse=True)

    def test_delete_checkpoint(self):
        """Test checkpoint deletion."""

        # Create and save checkpoint
        node = WorkflowNode(id="delete-test", type=WorkflowNodeType.VALIDATE, description="Delete test")
        plan = WorkflowPlan(id="delete-plan", source_vibe=Vibe(description="Delete test"), root_nodes=[node])
        result = ExecutionResult("delete-plan")

        checkpoint = WorkflowCheckpoint(plan, result)
        saved_path = self.persistence_manager.save_checkpoint(checkpoint)
        assert os.path.exists(saved_path)

        # Delete checkpoint
        deleted = self.persistence_manager.delete_checkpoint(checkpoint.checkpoint_id)
        assert deleted is True
        assert not os.path.exists(saved_path)

        # Try to delete non-existent checkpoint
        deleted_again = self.persistence_manager.delete_checkpoint(checkpoint.checkpoint_id)
        assert deleted_again is False

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint raises appropriate error."""

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            self.persistence_manager.load_checkpoint("nonexistent-checkpoint")

    def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted checkpoint files."""

        # Create corrupted checkpoint file
        corrupted_path = os.path.join(self.temp_dir, "corrupted_checkpoint.json")
        with open(corrupted_path, 'w') as f:
            f.write("{ invalid json }")

        # Should raise RuntimeError when trying to load
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            self.persistence_manager.load_checkpoint("corrupted_checkpoint")

        # List checkpoints should skip corrupted files
        checkpoints = self.persistence_manager.list_checkpoints()
        assert len(checkpoints) == 0  # Corrupted file should be skipped

    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory is created automatically."""

        new_temp_dir = os.path.join(self.temp_dir, "nested", "checkpoint", "dir")
        assert not os.path.exists(new_temp_dir)

        # Creating persistence manager should create directory
        manager = WorkflowPersistenceManager(new_temp_dir)
        assert os.path.exists(new_temp_dir)

    def test_json_serialization_format(self):
        """Test that saved JSON files have correct format and structure."""

        node = WorkflowNode(id="json-test", type=WorkflowNodeType.TRANSFORM, description="JSON format test")
        plan = WorkflowPlan(id="json-plan", source_vibe=Vibe(description="JSON test"), root_nodes=[node])
        result = ExecutionResult("json-plan")

        checkpoint = WorkflowCheckpoint(plan, result)
        saved_path = self.persistence_manager.save_checkpoint(checkpoint)

        # Read and verify JSON structure
        with open(saved_path, 'r') as f:
            data = json.load(f)

        # Verify top-level structure
        expected_keys = {"schema_version", "checkpoint_id", "created_at", "plan", "execution_result"}
        assert set(data.keys()) == expected_keys

        # Verify plan structure
        plan_data = data["plan"]
        assert "id" in plan_data
        assert "source_vibe" in plan_data
        assert "root_nodes" in plan_data

        # Verify execution result structure
        result_data = data["execution_result"]
        assert "plan_id" in result_data
        assert "status" in result_data
        assert "node_results" in result_data

        # Verify JSON is properly formatted (indented)
        with open(saved_path, 'r') as f:
            content = f.read()
        assert "  " in content  # Should have indentation

    def test_file_save_error_handling(self):
        """Test handling of file save errors."""

        # Create checkpoint
        node = WorkflowNode(id="error-test", type=WorkflowNodeType.GENERATE, description="Error test")
        plan = WorkflowPlan(id="error-plan", source_vibe=Vibe(description="Error test"), root_nodes=[node])
        result = ExecutionResult("error-plan")
        checkpoint = WorkflowCheckpoint(plan, result)

        # Use a valid directory but simulate write failure by creating a checkpoint with invalid file name
        import tempfile
        import unittest.mock
        valid_manager = WorkflowPersistenceManager(self.temp_dir)

        # Mock open to raise an exception
        with unittest.mock.patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should raise RuntimeError when save fails
            with pytest.raises(RuntimeError, match="Failed to save checkpoint"):
                valid_manager.save_checkpoint(checkpoint)