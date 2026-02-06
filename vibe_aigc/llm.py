"""LLM client abstraction for Vibe decomposition.

Supports multiple providers:
- OpenAI (OPENAI_API_KEY)
- Anthropic (ANTHROPIC_API_KEY)
- Ollama (local, no key needed - uses OpenAI-compatible API)
"""

import asyncio
import json
import os
from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

from .models import Vibe, WorkflowPlan


def _load_dotenv():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        # Try vibe-aigc root, then workspace root
        for path in [".env", "../.env"]:
            if os.path.exists(path):
                load_dotenv(path)
                break
    except ImportError:
        pass


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AUTO = "auto"  # Auto-detect based on available keys


# Default models per provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OLLAMA: "qwen2.5:14b",  # Good balance of speed/quality
}

# Ollama models known to work well for planning
OLLAMA_RECOMMENDED_MODELS = [
    "qwen2.5-coder:32b-instruct-q4_K_M",  # Best for structured output
    "glm-4.7-flash:latest",
    "qwen2.5:14b",
    "qwen2.5:7b",  # Faster, smaller
]


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    provider: LLMProvider = LLMProvider.AUTO
    model: Optional[str] = None  # None = use provider default
    temperature: float = 0.7
    max_tokens: int = 4000
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # Custom endpoint
    
    # Ollama-specific
    ollama_host: str = "http://localhost:11434"
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables with auto-detection."""
        _load_dotenv()
        
        # Check for explicit provider
        provider_str = os.getenv("LLM_PROVIDER", "auto").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.AUTO
        
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    
    @classmethod
    def for_ollama(cls, host: str = "http://localhost:11434", model: str = "qwen2.5:14b") -> "LLMConfig":
        """Convenience constructor for Ollama."""
        return cls(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url=f"{host.rstrip('/')}/v1",
            ollama_host=host,
        )
    
    @classmethod
    def for_openai(cls, api_key: Optional[str] = None, model: str = "gpt-4") -> "LLMConfig":
        """Convenience constructor for OpenAI."""
        return cls(
            provider=LLMProvider.OPENAI,
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    @classmethod
    def for_anthropic(cls, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514") -> "LLMConfig":
        """Convenience constructor for Anthropic."""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
    
    def resolve_provider(self) -> LLMProvider:
        """Resolve AUTO provider to actual provider based on available credentials."""
        if self.provider != LLMProvider.AUTO:
            return LLMProvider(self.provider)
        
        _load_dotenv()
        
        # Priority: explicit base_url > API keys > Ollama
        if self.base_url:
            # Custom endpoint - assume OpenAI-compatible
            return LLMProvider.OPENAI
        
        if self.api_key or os.getenv("OPENAI_API_KEY"):
            return LLMProvider.OPENAI
        
        if os.getenv("ANTHROPIC_API_KEY"):
            return LLMProvider.ANTHROPIC
        
        # Default to Ollama (no key needed)
        return LLMProvider.OLLAMA
    
    def get_model(self) -> str:
        """Get model name, using default if not specified."""
        if self.model:
            return self.model
        provider = self.resolve_provider()
        return DEFAULT_MODELS.get(provider, "gpt-4")


class LLMClient:
    """Async client for LLM-based Vibe decomposition.
    
    Supports OpenAI, Anthropic, and Ollama backends.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        # Load from env if no config provided
        if config is None:
            config = LLMConfig.from_env()
        self.config = config
        self.provider = config.resolve_provider()
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic_client()
        else:
            # OpenAI and Ollama both use OpenAI-compatible API
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI or Ollama client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "openai package required. Install with: pip install openai"
            )
        
        client_kwargs = {}
        
        if self.provider == LLMProvider.OLLAMA:
            # Ollama uses OpenAI-compatible API
            base_url = self.config.base_url or f"{self.config.ollama_host.rstrip('/')}/v1"
            client_kwargs["base_url"] = base_url
            client_kwargs["api_key"] = "ollama"  # Ollama doesn't need a real key
        else:
            # OpenAI or custom endpoint
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
        
        try:
            self._client = AsyncOpenAI(**client_kwargs)
        except Exception as e:
            if "api_key" in str(e).lower() and self.provider != LLMProvider.OLLAMA:
                raise RuntimeError(
                    f"OpenAI API key required. Set OPENAI_API_KEY or use Ollama:\n"
                    f"  LLMConfig.for_ollama('http://localhost:11434')\n"
                    f"Original error: {e}"
                ) from e
            raise
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package required. Install with: pip install anthropic"
            )
        
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key to config."
            )
        
        self._client = AsyncAnthropic(api_key=api_key)

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
Focus on logical decomposition and clear dependencies. Keep tasks atomic and executable.

Return ONLY valid JSON, no markdown code blocks or explanatory text."""

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

        # Dispatch to appropriate provider
        if self.provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(system_prompt, user_prompt)
        else:
            return await self._call_openai_compatible(system_prompt, user_prompt)

    async def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call OpenAI or Ollama (OpenAI-compatible API)."""
        model = self.config.get_model()
        
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            content = response.choices[0].message.content
            return self._parse_json_response(content)

        except Exception as e:
            return self._handle_error(e, model)
    
    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        model = self.config.get_model()
        
        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = response.content[0].text
            return self._parse_json_response(content)

        except Exception as e:
            return self._handle_error(e, model)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        if not content:
            raise ValueError(
                "Empty response from LLM. This could indicate an API issue or "
                "the request was filtered. Please try again or adjust your vibe."
            )

        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            # Remove opening ```json or ``` 
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            # Remove closing ```
            if content.endswith("```"):
                content = content[:-3].strip()
        
        # Try to find JSON object if there's extra text
        if not content.startswith("{"):
            start = content.find("{")
            if start != -1:
                end = content.rfind("}") + 1
                if end > start:
                    content = content[start:end]

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON response from LLM: {e}. "
                f"Response content: {content[:200]}..."
            ) from e
    
    def _handle_error(self, e: Exception, model: str) -> Dict[str, Any]:
        """Handle and re-raise errors with helpful messages."""
        error_lower = str(e).lower()
        
        if any(kw in error_lower for kw in ["api_key", "unauthorized", "authentication", "invalid.*key"]):
            raise RuntimeError(
                f"LLM authentication failed: {e}. "
                f"Provider: {self.provider.value}, Model: {model}\n"
                f"For local development, use Ollama: LLMConfig.for_ollama()"
            ) from e
        elif "rate limit" in error_lower:
            raise RuntimeError(
                f"API rate limit exceeded: {e}. "
                "Please wait a moment and try again."
            ) from e
        elif any(kw in error_lower for kw in ["timeout", "connection", "refused"]):
            if self.provider == LLMProvider.OLLAMA:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.config.ollama_host}: {e}\n"
                    f"Make sure Ollama is running: ollama serve"
                ) from e
            raise RuntimeError(
                f"Network error while calling LLM: {e}"
            ) from e
        elif "model" in error_lower and "not found" in error_lower:
            if self.provider == LLMProvider.OLLAMA:
                raise RuntimeError(
                    f"Model '{model}' not found in Ollama.\n"
                    f"Pull it with: ollama pull {model}\n"
                    f"Or use a different model: LLMConfig.for_ollama(model='qwen2.5:7b')"
                ) from e
            raise RuntimeError(f"Model '{model}' not available: {e}") from e
        else:
            raise RuntimeError(
                f"LLM request failed ({self.provider.value}/{model}): {e}"
            ) from e


async def list_ollama_models(host: str = "http://localhost:11434") -> List[str]:
    """List available models on an Ollama instance.
    
    Args:
        host: Ollama server URL
        
    Returns:
        List of model names
    """
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{host.rstrip('/')}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
    except Exception:
        return []


async def check_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama is available at the given host."""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{host.rstrip('/')}/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                return resp.status == 200
    except Exception:
        return False
