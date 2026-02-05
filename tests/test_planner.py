"""Tests for MetaPlanner and LLM integration."""

import pytest
import json
from unittest.mock import AsyncMock, patch

from vibe_aigc.models import Vibe, WorkflowNodeType
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.llm import LLMClient, LLMConfig


@pytest.mark.asyncio
class TestLLMClient:
    """Test LLM client functionality with mocked responses."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_decompose_vibe_success(self, mock_openai):
        """Test successful Vibe decomposition."""

        # Mock LLM response
        mock_response = {
            "id": "plan-test-001",
            "root_nodes": [
                {
                    "id": "task-001",
                    "type": "analyze",
                    "description": "Analyze scene requirements",
                    "parameters": {"detail_level": "high"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 30
                }
            ]
        }

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = json.dumps(mock_response)
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test decomposition
        client = LLMClient(LLMConfig())
        vibe = Vibe(description="Create a cinematic sci-fi scene")

        result = await client.decompose_vibe(vibe)

        assert result["id"] == "plan-test-001"
        assert len(result["root_nodes"]) == 1
        assert result["root_nodes"][0]["type"] == "analyze"

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_decompose_vibe_invalid_json(self, mock_openai):
        """Test handling of invalid JSON response."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig())
        vibe = Vibe(description="Test")

        with pytest.raises(ValueError, match="Invalid JSON response from LLM"):
            await client.decompose_vibe(vibe)


@pytest.mark.asyncio
class TestMetaPlanner:
    """Test MetaPlanner core functionality."""

    @patch('vibe_aigc.planner.LLMClient')
    async def test_plan_generation(self, mock_llm_client):
        """Test WorkflowPlan generation from Vibe."""

        # Mock LLM client response
        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "id": "plan-test-001",
            "root_nodes": [
                {
                    "id": "task-001",
                    "type": "generate",
                    "description": "Generate sci-fi concept",
                    "parameters": {"style": "cinematic"},
                    "dependencies": [],
                    "children": [
                        {
                            "id": "subtask-001",
                            "type": "analyze",
                            "description": "Analyze style requirements",
                            "parameters": {},
                            "dependencies": [],
                            "children": [],
                            "estimated_duration": 15
                        }
                    ],
                    "estimated_duration": 60
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        # Test plan generation
        planner = MetaPlanner()
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        plan = await planner.plan(vibe)

        assert plan.id == "plan-test-001"
        assert plan.source_vibe.description == "Create a cinematic sci-fi scene"
        assert len(plan.root_nodes) == 1
        assert plan.root_nodes[0].type == WorkflowNodeType.GENERATE
        assert len(plan.root_nodes[0].children) == 1
        assert plan.estimated_total_duration == 75  # 60 + 15

    @patch('vibe_aigc.planner.LLMClient')
    async def test_execute_basic_workflow(self, mock_llm_client):
        """Test basic execute method (Phase 2 implementation)."""

        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "root_nodes": [
                {
                    "id": "task-001",
                    "type": "generate",
                    "description": "Test task",
                    "estimated_duration": 30
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()
        vibe = Vibe(description="Test vibe")

        result = await planner.execute(vibe)

        assert result["status"] == "completed"
        assert result["vibe_description"] == "Test vibe"

        # Check execution summary
        execution_summary = result["execution_summary"]
        assert execution_summary["total_nodes"] == 1
        assert execution_summary["completed"] == 1
        assert execution_summary["failed"] == 0

        # Check node results
        node_results = result["node_results"]
        assert "task-001" in node_results
        assert node_results["task-001"]["status"] == "completed"

    @patch('vibe_aigc.planner.LLMClient')
    async def test_node_type_mapping_and_validation(self, mock_llm_client):
        """Test node type mapping and parameter preservation."""

        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "id": "plan-node-types",
            "root_nodes": [
                {
                    "id": "analyze-task",
                    "type": "analyze",  # Should map to WorkflowNodeType.ANALYZE
                    "description": "Analyze user requirements",
                    "parameters": {"depth": "comprehensive", "focus": "constraints"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 45
                },
                {
                    "id": "invalid-type-task",
                    "type": "unknown_type",  # Should default to GENERATE
                    "description": "Task with invalid type",
                    "parameters": {"fallback": True},
                    "dependencies": ["analyze-task"],
                    "children": [],
                    "estimated_duration": 30
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()
        vibe = Vibe(description="Test node type mapping")

        plan = await planner.plan(vibe)

        # Verify node type mapping
        analyze_node = plan.root_nodes[0]
        assert analyze_node.type == WorkflowNodeType.ANALYZE
        assert analyze_node.parameters == {"depth": "comprehensive", "focus": "constraints"}
        assert analyze_node.dependencies == []

        # Verify fallback for invalid types
        invalid_node = plan.root_nodes[1]
        assert invalid_node.type == WorkflowNodeType.GENERATE  # Should default
        assert invalid_node.dependencies == ["analyze-task"]
        assert invalid_node.parameters == {"fallback": True}

        # Verify duration calculation
        assert plan.estimated_total_duration == 75  # 45 + 30

    @patch('vibe_aigc.planner.LLMClient')
    async def test_hierarchical_duration_calculation(self, mock_llm_client):
        """Test recursive duration calculation for nested nodes."""

        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "root_nodes": [
                {
                    "id": "parent-task",
                    "type": "composite",
                    "description": "Parent task with children",
                    "parameters": {},
                    "dependencies": [],
                    "children": [
                        {
                            "id": "child-1",
                            "type": "analyze",
                            "description": "Child task 1",
                            "parameters": {},
                            "dependencies": [],
                            "children": [
                                {
                                    "id": "grandchild-1",
                                    "type": "validate",
                                    "description": "Grandchild task",
                                    "parameters": {},
                                    "dependencies": [],
                                    "children": [],
                                    "estimated_duration": 10
                                }
                            ],
                            "estimated_duration": 20
                        },
                        {
                            "id": "child-2",
                            "type": "generate",
                            "description": "Child task 2",
                            "parameters": {},
                            "dependencies": ["child-1"],
                            "children": [],
                            "estimated_duration": 30
                        }
                    ],
                    "estimated_duration": 40
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()
        vibe = Vibe(description="Test hierarchical duration")

        plan = await planner.plan(vibe)

        # Total should be 40 (parent) + 20 (child-1) + 10 (grandchild) + 30 (child-2) = 100
        assert plan.estimated_total_duration == 100

        # Verify hierarchical structure is preserved
        parent = plan.root_nodes[0]
        assert len(parent.children) == 2
        assert parent.children[0].id == "child-1"
        assert len(parent.children[0].children) == 1
        assert parent.children[0].children[0].id == "grandchild-1"