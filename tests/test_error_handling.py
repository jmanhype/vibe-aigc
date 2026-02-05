"""Tests for comprehensive error handling across the Vibe AIGC system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from vibe_aigc.models import Vibe
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.llm import LLMClient, LLMConfig


@pytest.mark.asyncio
class TestLLMErrorHandling:
    """Test LLM client error handling with descriptive messages."""

    def test_missing_api_key_setup_guidance(self):
        """Test helpful error message for missing API key."""

        with pytest.raises(RuntimeError, match="OpenAI API key is required"):
            LLMClient(LLMConfig())

        with pytest.raises(RuntimeError, match="https://platform.openai.com/api-keys"):
            LLMClient(LLMConfig(api_key=None))

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_empty_llm_response_guidance(self, mock_openai):
        """Test handling of empty LLM response with helpful message."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig(api_key="test-key"))
        vibe = Vibe(description="Test empty response")

        with pytest.raises(ValueError) as exc_info:
            await client.decompose_vibe(vibe)

        error_msg = str(exc_info.value)
        assert "Empty response from LLM" in error_msg
        assert "API issue or" in error_msg and "filtered" in error_msg

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_invalid_json_response_with_context(self, mock_openai):
        """Test handling of malformed JSON with response content."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = "Not valid JSON at all"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig(api_key="test-key"))
        vibe = Vibe(description="Test malformed JSON")

        with pytest.raises(ValueError, match="Invalid JSON response from LLM"):
            await client.decompose_vibe(vibe)

        with pytest.raises(ValueError, match="Response content: Not valid JSON"):
            await client.decompose_vibe(vibe)

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_authentication_error_guidance(self, mock_openai):
        """Test handling of authentication errors with API key guidance."""

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Unauthorized: Invalid API key")
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig(api_key="test-key"))
        vibe = Vibe(description="Test auth error")

        with pytest.raises(RuntimeError, match="LLM authentication failed"):
            await client.decompose_vibe(vibe)

        with pytest.raises(RuntimeError, match="check your OpenAI API key"):
            await client.decompose_vibe(vibe)

        with pytest.raises(RuntimeError, match="https://platform.openai.com/api-keys"):
            await client.decompose_vibe(vibe)

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_rate_limit_error_guidance(self, mock_openai):
        """Test handling of rate limit errors with helpful guidance."""

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig(api_key="test-key"))
        vibe = Vibe(description="Test rate limit")

        with pytest.raises(RuntimeError, match="API rate limit exceeded"):
            await client.decompose_vibe(vibe)

        with pytest.raises(RuntimeError, match="wait a moment and try again"):
            await client.decompose_vibe(vibe)

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_network_timeout_guidance(self, mock_openai):
        """Test handling of network timeouts with helpful message."""

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Request timeout after 30 seconds")
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig(api_key="test-key"))
        vibe = Vibe(description="Test timeout")

        with pytest.raises(RuntimeError, match="Network timeout while calling LLM"):
            await client.decompose_vibe(vibe)

        with pytest.raises(RuntimeError, match="check your internet connection"):
            await client.decompose_vibe(vibe)

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_generic_error_with_context(self, mock_openai):
        """Test handling of generic errors with helpful context."""

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Unexpected server error 500")
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig(api_key="test-key"))
        vibe = Vibe(description="Test generic error")

        with pytest.raises(RuntimeError, match="LLM request failed"):
            await client.decompose_vibe(vibe)

        with pytest.raises(RuntimeError, match="network issue, API outage, or configuration problem"):
            await client.decompose_vibe(vibe)


@pytest.mark.asyncio
class TestMetaPlannerErrorHandling:
    """Test MetaPlanner error handling and propagation."""

    @patch('vibe_aigc.planner.LLMClient')
    async def test_planning_failure_context(self, mock_llm_client):
        """Test that planning failures include vibe context."""

        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.side_effect = ValueError("JSON parsing failed")
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()
        vibe = Vibe(description="Test planning failure context")

        with pytest.raises(RuntimeError, match="Failed to generate workflow plan"):
            await planner.plan(vibe)

        with pytest.raises(RuntimeError, match="Test planning failure context"):
            await planner.plan(vibe)

    @patch('vibe_aigc.planner.LLMClient')
    async def test_execution_planning_failure_context(self, mock_llm_client):
        """Test that execution failures during planning include context."""

        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.side_effect = RuntimeError("API authentication failed")
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()
        vibe = Vibe(description="Test execution planning failure")

        with pytest.raises(RuntimeError, match="Failed to plan workflow for vibe"):
            await planner.execute(vibe)

        with pytest.raises(RuntimeError, match="Test execution planning failure"):
            await planner.execute(vibe)

    @patch('vibe_aigc.planner.LLMClient')
    async def test_execution_engine_failure_context(self, mock_llm_client):
        """Test that execution engine failures include plan context."""

        # Mock successful planning but failed execution
        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "id": "test-plan-failure",
            "root_nodes": [
                {
                    "id": "failing-task",
                    "type": "generate",
                    "description": "Task that will cause execution failure",
                    "parameters": {},
                    "dependencies": [],
                    "children": []
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()

        # Mock the executor to fail during execution
        mock_executor = MagicMock()
        mock_executor.execute_plan = AsyncMock(side_effect=Exception("Execution engine crashed"))
        planner.executor = mock_executor

        vibe = Vibe(description="Test execution engine failure")

        with pytest.raises(RuntimeError, match="Failed to execute workflow plan"):
            await planner.execute(vibe)

        with pytest.raises(RuntimeError, match="test-plan-failure"):
            await planner.execute(vibe)


@pytest.mark.asyncio
class TestEndToEndErrorPropagation:
    """Test error propagation through the entire system."""

    def test_configuration_error_chain(self):
        """Test that configuration errors propagate with helpful messages."""

        # Test missing API key propagation
        with pytest.raises(RuntimeError, match="OpenAI API key is required"):
            planner = MetaPlanner()  # Should fail during LLM client creation

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_full_chain_error_propagation(self, mock_openai):
        """Test error propagation from LLM through to user interface."""

        # Mock an authentication failure at the OpenAI level
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid authentication credentials")
        mock_openai.return_value = mock_client

        planner = MetaPlanner(LLMConfig(api_key="invalid-test-key"))
        vibe = Vibe(description="Test full chain error propagation")

        # The error should propagate through:
        # LLMClient.decompose_vibe -> MetaPlanner.plan -> MetaPlanner.execute

        try:
            await planner.execute(vibe)
            assert False, "Should have raised an error"
        except RuntimeError as e:
            error_msg = str(e)

            # Should contain context about the planning failure
            assert "Failed to plan workflow for vibe" in error_msg
            assert "Test full chain error propagation" in error_msg

            # Should chain to the LLM authentication error
            assert e.__cause__ is not None
            cause_msg = str(e.__cause__)
            assert "Failed to generate workflow plan" in cause_msg

            # Should ultimately chain to the LLM client error
            assert e.__cause__.__cause__ is not None
            root_cause_msg = str(e.__cause__.__cause__)
            assert "LLM authentication failed" in root_cause_msg
            assert "check your OpenAI API key" in root_cause_msg


@pytest.mark.asyncio
class TestUserFriendlyErrorMessages:
    """Test that error messages provide actionable guidance."""

    def test_setup_error_messages(self):
        """Test that setup errors provide clear next steps."""

        try:
            LLMClient()
        except RuntimeError as e:
            error_msg = str(e)
            assert "OPENAI_API_KEY environment variable" in error_msg
            assert "https://platform.openai.com" in error_msg
            assert "api_key to LLMConfig" in error_msg

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_runtime_error_messages(self, mock_openai):
        """Test that runtime errors suggest concrete actions."""

        test_cases = [
            {
                "error": "Rate limit exceeded for organization",
                "should_contain": ["wait a moment", "API plan limits"]
            },
            {
                "error": "Connection timeout after 30 seconds",
                "should_contain": ["internet connection", "try again"]
            },
            {
                "error": "Invalid authentication credentials",
                "should_contain": ["API key", "platform.openai.com"]
            }
        ]

        for case in test_cases:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception(case["error"])
            mock_openai.return_value = mock_client

            client = LLMClient(LLMConfig(api_key="test"))
            vibe = Vibe(description="Test error guidance")

            try:
                await client.decompose_vibe(vibe)
                assert False, f"Should have raised error for: {case['error']}"
            except RuntimeError as e:
                error_msg = str(e).lower()
                for expected_phrase in case["should_contain"]:
                    assert expected_phrase.lower() in error_msg, (
                        f"Expected '{expected_phrase}' in error message for '{case['error']}'"
                    )