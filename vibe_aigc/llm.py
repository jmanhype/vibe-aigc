"""LLM client abstraction for Vibe decomposition."""

import asyncio
import json
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

from .models import Vibe, WorkflowPlan


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None


class LLMClient:
    """Async client for LLM-based Vibe decomposition."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        try:
            self.client = AsyncOpenAI(api_key=self.config.api_key)
        except Exception as e:
            if "api_key" in str(e).lower():
                raise RuntimeError(
                    "OpenAI API key is required. Please set the OPENAI_API_KEY environment "
                    "variable or pass an api_key to LLMConfig. "
                    "Get your API key from: https://platform.openai.com/api-keys"
                ) from e
            raise

    async def decompose_vibe(self, vibe: Vibe) -> Dict[str, Any]:
        """Decompose a Vibe into structured workflow plan data."""

        system_prompt = """You are a Meta-Planner that decomposes high-level creative intent (Vibes) into executable workflow plans.

Given a Vibe, create a hierarchical breakdown of tasks needed to achieve the user's intent.

Respond with a JSON object containing:
- id: unique plan identifier
- root_nodes: array of top-level tasks, each with:
  - id: unique task identifier
  - type: one of "analyze", "generate", "transform", "validate", "composite"
  - description: clear task description
  - parameters: task-specific configuration
  - dependencies: array of task IDs that must complete first
  - children: array of sub-tasks (same structure)
  - estimated_duration: estimated seconds to complete

Focus on logical decomposition and clear dependencies. Keep tasks atomic and executable."""

        user_prompt = f"""Decompose this Vibe into a workflow plan:

Description: {vibe.description}
Style: {vibe.style or 'Not specified'}
Constraints: {', '.join(vibe.constraints) if vibe.constraints else 'None'}
Domain: {vibe.domain or 'General'}

Additional context: {vibe.metadata}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError(
                    "Empty response from LLM. This could indicate an API issue or "
                    "the request was filtered. Please try again or adjust your vibe."
                )

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON response from LLM: {e}. "
                f"The LLM returned malformed data. Please try again. "
                f"Response content: {content[:200] if 'content' in locals() else 'N/A'}..."
            ) from e
        except Exception as e:
            error_lower = str(e).lower()
            if any(keyword in error_lower for keyword in ["api_key", "unauthorized", "authentication", "invalid.*key"]):
                raise RuntimeError(
                    f"LLM authentication failed: {e}. "
                    "Please check your OpenAI API key and ensure it's valid. "
                    "Get your API key from: https://platform.openai.com/api-keys"
                ) from e
            elif "rate limit" in str(e).lower():
                raise RuntimeError(
                    f"OpenAI API rate limit exceeded: {e}. "
                    "Please wait a moment and try again, or check your API plan limits."
                ) from e
            elif "timeout" in str(e).lower():
                raise RuntimeError(
                    f"Network timeout while calling LLM: {e}. "
                    "Please check your internet connection and try again."
                ) from e
            else:
                raise RuntimeError(
                    f"LLM request failed: {e}. "
                    f"This could be a network issue, API outage, or configuration problem."
                ) from e