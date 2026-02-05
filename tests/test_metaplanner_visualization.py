"""Test MetaPlanner integration with visualization functionality."""

import pytest
from unittest.mock import patch, AsyncMock

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.visualization import VisualizationFormat
from vibe_aigc.executor import ProgressEvent, ProgressEventType


class TestMetaPlannerVisualization:
    """Test MetaPlanner visualization integration functionality."""

    @patch('vibe_aigc.planner.LLMClient')
    def test_constructor_with_progress_callback(self, mock_llm_client):
        """Test MetaPlanner constructor accepts progress callback parameter."""

        mock_client = AsyncMock()
        mock_llm_client.return_value = mock_client

        progress_events = []
        def test_callback(event: ProgressEvent):
            progress_events.append(event)

        # Should not raise with progress callback
        planner = MetaPlanner(progress_callback=test_callback)
        assert planner.progress_callback == test_callback
        assert planner.executor.progress_callback == test_callback

    @pytest.mark.asyncio
    @patch('vibe_aigc.planner.LLMClient')
    async def test_execute_with_visualization_ascii(self, mock_llm_client):
        """Test MetaPlanner.execute_with_visualization() with ASCII format."""

        # Mock LLM response
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "viz-test-plan",
            "root_nodes": [
                {
                    "id": "viz-node",
                    "type": "generate",
                    "description": "Test visualization node",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                }
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create MetaPlanner
        planner = MetaPlanner()
        vibe = Vibe(description="Test visualization integration")

        # Execute with visualization
        result = await planner.execute_with_visualization(vibe, visualization_format=VisualizationFormat.ASCII)

        # Verify result structure
        assert "status" in result
        assert "plan_id" in result
        assert "vibe_description" in result
        assert "visualization" in result
        assert "execution_summary" in result
        assert "node_results" in result

        # Verify visualization content
        visualization = result["visualization"]
        assert "Workflow Plan: viz-test-plan" in visualization
        assert "Test visualization integration" in visualization
        assert "Test visualization node" in visualization
        assert "generate" in visualization

    @pytest.mark.asyncio
    @patch('vibe_aigc.planner.LLMClient')
    async def test_execute_with_visualization_mermaid(self, mock_llm_client):
        """Test MetaPlanner.execute_with_visualization() with Mermaid format."""

        # Mock LLM response
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "mermaid-test-plan",
            "root_nodes": [
                {
                    "id": "mermaid-node",
                    "type": "analyze",
                    "description": "Test Mermaid node",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                }
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create MetaPlanner
        planner = MetaPlanner()
        vibe = Vibe(description="Test Mermaid visualization")

        # Execute with Mermaid visualization
        result = await planner.execute_with_visualization(vibe, visualization_format=VisualizationFormat.MERMAID)

        # Verify result structure
        assert "visualization" in result
        visualization = result["visualization"]

        # Verify Mermaid syntax
        assert "graph TD" in visualization
        assert "mermaid-node" in visualization
        assert "Test Mermaid node" in visualization
        assert "classDef" in visualization  # CSS styling

    @pytest.mark.asyncio
    @patch('vibe_aigc.planner.LLMClient')
    async def test_progress_callback_integration(self, mock_llm_client):
        """Test MetaPlanner progress callback integration."""

        # Mock LLM response
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "progress-test-plan",
            "root_nodes": [
                {
                    "id": "progress-node",
                    "type": "generate",
                    "description": "Test progress tracking",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                }
            ]
        }
        mock_llm_client.return_value = mock_client

        # Capture progress events
        progress_events = []
        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        # Create MetaPlanner with progress callback
        planner = MetaPlanner(progress_callback=capture_progress)
        vibe = Vibe(description="Test progress integration")

        # Execute with visualization
        result = await planner.execute_with_visualization(vibe, show_progress=False)  # Don't override callback

        # Verify progress events were captured
        assert len(progress_events) > 0

        event_types = [event.event_type for event in progress_events]
        assert ProgressEventType.WORKFLOW_STARTED in event_types
        assert ProgressEventType.WORKFLOW_COMPLETED in event_types

        # Verify result still includes visualization
        assert "visualization" in result

    @pytest.mark.asyncio
    @patch('vibe_aigc.planner.LLMClient')
    async def test_parallel_efficiency_in_visualization_result(self, mock_llm_client):
        """Test that parallel efficiency metrics are included in visualization result."""

        # Mock LLM response with multiple nodes
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "parallel-viz-plan",
            "root_nodes": [
                {
                    "id": "parallel-1",
                    "type": "generate",
                    "description": "First parallel task",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                },
                {
                    "id": "parallel-2",
                    "type": "generate",
                    "description": "Second parallel task",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                }
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create MetaPlanner
        planner = MetaPlanner()
        vibe = Vibe(description="Test parallel visualization")

        # Execute with visualization
        result = await planner.execute_with_visualization(vibe)

        # Verify parallel metrics are included
        assert "parallel_efficiency" in result
        assert "execution_groups" in result
        assert isinstance(result["parallel_efficiency"], float)
        assert isinstance(result["execution_groups"], list)

        # Verify visualization shows parallel efficiency
        visualization = result["visualization"]
        assert "Parallel Efficiency:" in visualization

    @pytest.mark.asyncio
    @patch('vibe_aigc.planner.LLMClient')
    async def test_backward_compatibility_with_execute(self, mock_llm_client):
        """Test that existing execute() method still works (backward compatibility)."""

        # Mock LLM response
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "compat-test-plan",
            "root_nodes": [
                {
                    "id": "compat-node",
                    "type": "analyze",
                    "description": "Compatibility test",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                }
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create MetaPlanner
        planner = MetaPlanner()
        vibe = Vibe(description="Test backward compatibility")

        # Old execute method should still work
        result = await planner.execute(vibe)

        # Should have the original result format (no visualization key)
        assert "status" in result
        assert "plan_id" in result
        assert "visualization" not in result  # Old method doesn't include visualization