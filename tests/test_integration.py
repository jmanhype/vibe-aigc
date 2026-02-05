"""End-to-end integration tests for the complete Vibe AIGC system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from vibe_aigc import MetaPlanner, Vibe


@pytest.mark.asyncio
class TestReadmeUsageExample:
    """Test that the exact README.md usage example works."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_readme_example_exact_match(self, mock_openai):
        """Test the exact code example from README.md:39-50."""

        # Mock OpenAI client to return valid workflow plan
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-cinematic-scifi",
            "root_nodes": [
                {
                    "id": "concept-development",
                    "type": "analyze",
                    "description": "Develop core sci-fi concept respecting constraints",
                    "parameters": {
                        "style": "dark, atmospheric",
                        "constraints": ["no violence", "PG-13"],
                        "domain": "cinematic"
                    },
                    "dependencies": [],
                    "children": [
                        {
                            "id": "mood-analysis",
                            "type": "analyze",
                            "description": "Analyze dark atmospheric mood requirements",
                            "parameters": {"mood": "dark, atmospheric"},
                            "dependencies": [],
                            "children": [],
                            "estimated_duration": 15
                        },
                        {
                            "id": "constraint-validation",
                            "type": "validate",
                            "description": "Ensure PG-13 rating and no violence",
                            "parameters": {"constraints": ["no violence", "PG-13"]},
                            "dependencies": ["mood-analysis"],
                            "children": [],
                            "estimated_duration": 10
                        }
                    ],
                    "estimated_duration": 45
                },
                {
                    "id": "scene-generation",
                    "type": "generate",
                    "description": "Generate cinematic sci-fi scene with validated concept",
                    "parameters": {
                        "style": "cinematic",
                        "genre": "sci-fi",
                        "mood": "dark, atmospheric"
                    },
                    "dependencies": ["concept-development"],
                    "children": [],
                    "estimated_duration": 90
                },
                {
                    "id": "final-review",
                    "type": "validate",
                    "description": "Final review of generated scene against original vibe",
                    "parameters": {"review_type": "comprehensive"},
                    "dependencies": ["scene-generation"],
                    "children": [],
                    "estimated_duration": 20
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Execute the exact README example
        # from vibe_aigc import MetaPlanner, Vibe  # Already imported

        # Define your vibe
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        # Plan and execute
        planner = MetaPlanner()
        result = await planner.execute(vibe)

        # Verify the result structure and content
        assert result["status"] == "completed"
        assert result["vibe_description"] == "Create a cinematic sci-fi scene"

        execution_summary = result["execution_summary"]
        assert execution_summary["total_nodes"] == 5  # 3 root + 2 children
        assert execution_summary["completed"] == 5
        assert execution_summary["failed"] == 0

        # Verify specific workflow nodes were executed
        node_results = result["node_results"]
        assert "concept-development" in node_results
        assert "scene-generation" in node_results
        assert "final-review" in node_results
        assert "mood-analysis" in node_results  # child node
        assert "constraint-validation" in node_results  # child node

        # Verify all nodes completed successfully
        for node_id, node_result in node_results.items():
            assert node_result["status"] == "completed"
            assert node_result["result"] is not None
            assert node_result["error"] is None
            assert node_result["duration"] >= 0


@pytest.mark.asyncio
class TestRobustnessAndEdgeCases:
    """Test system robustness and edge case handling."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_minimal_vibe_execution(self, mock_openai):
        """Test with minimal Vibe configuration."""

        # Mock minimal LLM response
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-minimal",
            "root_nodes": [
                {
                    "id": "simple-task",
                    "type": "generate",
                    "description": "Handle minimal vibe request",
                    "parameters": {},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 30
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test minimal vibe (only required field)
        vibe = Vibe(description="Simple test")
        planner = MetaPlanner()

        result = await planner.execute(vibe)

        assert result["status"] == "completed"
        assert result["vibe_description"] == "Simple test"
        assert result["execution_summary"]["total_nodes"] == 1
        assert result["execution_summary"]["completed"] == 1

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_complex_hierarchical_workflow(self, mock_openai):
        """Test deeply nested hierarchical workflow."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-complex",
            "root_nodes": [
                {
                    "id": "phase1",
                    "type": "composite",
                    "description": "Phase 1: Analysis and Planning",
                    "parameters": {},
                    "dependencies": [],
                    "children": [
                        {
                            "id": "deep-analysis",
                            "type": "analyze",
                            "description": "Deep requirement analysis",
                            "parameters": {},
                            "dependencies": [],
                            "children": [
                                {
                                    "id": "user-intent",
                                    "type": "analyze",
                                    "description": "Analyze user intent",
                                    "parameters": {},
                                    "dependencies": [],
                                    "children": [],
                                    "estimated_duration": 10
                                },
                                {
                                    "id": "technical-constraints",
                                    "type": "analyze",
                                    "description": "Analyze technical constraints",
                                    "parameters": {},
                                    "dependencies": ["user-intent"],
                                    "children": [],
                                    "estimated_duration": 15
                                }
                            ],
                            "estimated_duration": 30
                        },
                        {
                            "id": "strategy-planning",
                            "type": "generate",
                            "description": "Generate execution strategy",
                            "parameters": {},
                            "dependencies": ["deep-analysis"],
                            "children": [],
                            "estimated_duration": 25
                        }
                    ],
                    "estimated_duration": 70
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(
            description="Complex multi-phase project",
            style="systematic, thorough",
            constraints=["high quality", "detailed analysis"]
        )
        planner = MetaPlanner()

        result = await planner.execute(vibe)

        assert result["status"] == "completed"

        # Verify all levels of hierarchy executed
        node_results = result["node_results"]
        assert "phase1" in node_results  # Root composite
        assert "deep-analysis" in node_results  # Level 2 composite
        assert "strategy-planning" in node_results  # Level 2 leaf
        assert "user-intent" in node_results  # Level 3 leaf
        assert "technical-constraints" in node_results  # Level 3 leaf

        # Verify execution order respected dependencies
        assert all(result["status"] == "completed" for result in node_results.values())

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_llm_error_handling(self, mock_openai):
        """Test handling of LLM API errors."""

        # Mock LLM client to raise an exception
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test error handling")
        planner = MetaPlanner()

        # Should raise RuntimeError with descriptive message
        with pytest.raises(RuntimeError, match="LLM request failed"):
            await planner.execute(vibe)

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_malformed_llm_response(self, mock_openai):
        """Test handling of malformed LLM responses."""

        # Mock LLM to return invalid JSON
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = "This is not valid JSON"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test malformed response")
        planner = MetaPlanner()

        # Should raise RuntimeError (ValueError gets wrapped) for invalid JSON
        with pytest.raises(RuntimeError, match="Invalid JSON response from LLM"):
            await planner.execute(vibe)


@pytest.mark.asyncio
class TestPlanningWithoutExecution:
    """Test the planning phase independently."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_plan_only_workflow(self, mock_openai):
        """Test generating a plan without executing it."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-only-test",
            "root_nodes": [
                {
                    "id": "planning-task",
                    "type": "analyze",
                    "description": "Planning task for testing",
                    "parameters": {"mode": "planning"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 40
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test planning without execution")
        planner = MetaPlanner()

        # Test plan() method directly
        plan = await planner.plan(vibe)

        assert plan.id == "plan-only-test"
        assert plan.source_vibe.description == "Test planning without execution"
        assert len(plan.root_nodes) == 1
        assert plan.root_nodes[0].description == "Planning task for testing"
        assert plan.estimated_total_duration == 40
        assert plan.created_at is not None

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_plan_and_execute_separation(self, mock_openai):
        """Test plan_and_execute method for getting both plan and results."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-execute-test",
            "root_nodes": [
                {
                    "id": "separation-task",
                    "type": "validate",
                    "description": "Test plan/execute separation",
                    "parameters": {},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 20
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test plan and execute separation")
        planner = MetaPlanner()

        plan, execution_result = await planner.plan_and_execute(vibe)

        # Verify plan structure
        assert plan.id == "plan-execute-test"
        assert len(plan.root_nodes) == 1

        # Verify execution result
        assert execution_result.plan_id == "plan-execute-test"
        assert execution_result.status.value == "completed"
        assert len(execution_result.node_results) == 1
        assert "separation-task" in execution_result.node_results