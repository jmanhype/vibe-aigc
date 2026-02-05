"""Tests for core data models."""

import pytest
from vibe_aigc.models import Vibe, WorkflowNode, WorkflowPlan, WorkflowNodeType


class TestVibe:
    """Test Vibe model validation and behavior."""

    def test_basic_vibe_creation(self):
        """Test creating a basic Vibe with required fields."""
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        assert vibe.description == "Create a cinematic sci-fi scene"
        assert vibe.style == "dark, atmospheric"
        assert vibe.constraints == ["no violence", "PG-13"]
        assert vibe.domain is None
        assert vibe.metadata == {}

    def test_vibe_with_optional_fields(self):
        """Test Vibe creation with all optional fields."""
        vibe = Vibe(
            description="Test description",
            domain="visual",
            metadata={"quality": "high", "format": "4K"}
        )

        assert vibe.domain == "visual"
        assert vibe.metadata == {"quality": "high", "format": "4K"}

    def test_vibe_validation_requires_description(self):
        """Test that description field is required."""
        with pytest.raises(ValueError):
            Vibe()


class TestWorkflowNode:
    """Test WorkflowNode model validation and hierarchical structure."""

    def test_basic_node_creation(self):
        """Test creating a basic WorkflowNode."""
        node = WorkflowNode(
            id="task-001",
            type=WorkflowNodeType.GENERATE,
            description="Generate initial concept"
        )

        assert node.id == "task-001"
        assert node.type == WorkflowNodeType.GENERATE
        assert node.description == "Generate initial concept"
        assert node.parameters == {}
        assert node.dependencies == []
        assert node.children == []

    def test_hierarchical_node_structure(self):
        """Test creating nodes with children for hierarchical decomposition."""
        child1 = WorkflowNode(
            id="subtask-001",
            type=WorkflowNodeType.ANALYZE,
            description="Analyze requirements"
        )
        child2 = WorkflowNode(
            id="subtask-002",
            type=WorkflowNodeType.GENERATE,
            description="Generate content",
            dependencies=["subtask-001"]
        )

        parent = WorkflowNode(
            id="task-001",
            type=WorkflowNodeType.COMPOSITE,
            description="Complete content creation",
            children=[child1, child2]
        )

        assert len(parent.children) == 2
        assert parent.children[0].id == "subtask-001"
        assert parent.children[1].dependencies == ["subtask-001"]


class TestWorkflowPlan:
    """Test WorkflowPlan model and plan structure."""

    def test_basic_plan_creation(self):
        """Test creating a WorkflowPlan with minimal structure."""
        vibe = Vibe(description="Test vibe")
        node = WorkflowNode(
            id="task-001",
            type=WorkflowNodeType.GENERATE,
            description="Test task"
        )

        plan = WorkflowPlan(
            id="plan-001",
            source_vibe=vibe,
            root_nodes=[node]
        )

        assert plan.id == "plan-001"
        assert plan.source_vibe.description == "Test vibe"
        assert len(plan.root_nodes) == 1
        assert plan.root_nodes[0].id == "task-001"