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
    base_url: Optional[str] = None  # Custom endpoint (e.g., z.ai, local models)


class LLMClient:
    """Async client for LLM-based Vibe decomposition."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        try:
            client_kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self.client = AsyncOpenAI(**client_kwargs)
        except Exception as e:
            if "api_key" in str(e).lower():
                raise RuntimeError(
                    "OpenAI API key is required. Please set the OPENAI_API_KEY environment "
                    "variable or pass an api_key to LLMConfig. "
                    "Get your API key from: https://platform.openai.com/api-keys"
                ) from e
            raise

    async def decompose_vibe(
        self, 
        vibe: Vibe,
        knowledge_context: Optional[str] = None,
        tools_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Decompose a Vibe into structured workflow plan data.
        
        This implements the paper's MetaPlanner (Section 5.2):
        - Receives Vibe as input
        - Uses domain knowledge for intent understanding
        - Maps to available tools for execution
        
        Args:
            vibe: The high-level creative intent
            knowledge_context: Domain expertise for interpreting the vibe
            tools_context: Available tools for execution
        """

        system_prompt = """You are a Meta-Planner that decomposes high-level creative intent (Vibes) into executable workflow plans.

Given a Vibe, create a hierarchical breakdown of tasks needed to achieve the user's intent.

Respond with a JSON object containing:
- id: unique plan identifier
- root_nodes: array of top-level tasks, each with:
  - id: unique task identifier
  - type: one of "analyze", "generate", "transform", "validate", "composite"
  - description: clear task description
  - parameters: task-specific configuration including:
    - tool: name of the tool to use for execution (from available tools)
    - tool_inputs: inputs to pass to the tool
  - dependencies: array of task IDs that must complete first
  - children: array of sub-tasks (same structure)
  - estimated_duration: estimated seconds to complete

IMPORTANT: Each node should specify which tool to use for execution. Use the available tools provided.
Focus on logical decomposition and clear dependencies. Keep tasks atomic and executable."""

        # Build user prompt with context
        user_prompt_parts = [
            f"Decompose this Vibe into a workflow plan:",
            f"",
            f"Description: {vibe.description}",
            f"Style: {vibe.style or 'Not specified'}",
            f"Constraints: {', '.join(vibe.constraints) if vibe.constraints else 'None'}",
            f"Domain: {vibe.domain or 'General'}",
            f"",
            f"Additional context: {vibe.metadata}"
        ]
        
        # Add knowledge context (Paper Section 5.3)
        if knowledge_context:
            user_prompt_parts.extend([
                "",
                "---",
                knowledge_context
            ])
        
        # Add tools context (Paper Section 5.4)
        if tools_context:
            user_prompt_parts.extend([
                "",
                "---",
                tools_context,
                "",
                "Use the available tools above when specifying how each node should be executed."
            ])
        
        user_prompt = "\n".join(user_prompt_parts)

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

            # Strip markdown code blocks if present (common with some LLMs like z.ai/GLM)
            content = content.strip()
            if content.startswith("```"):
                # Remove opening ```json or ``` 
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1:]
                # Remove closing ```
                if content.endswith("```"):
                    content = content[:-3].strip()

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