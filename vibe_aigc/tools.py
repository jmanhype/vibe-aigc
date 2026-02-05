"""
Atomic Tool Library - Content Generation Tools.

As defined in Paper Section 5.4:
"The Planner traverses the system's atomic tool library—which includes
various Agents, foundation models, and media processing modules—to
select the optimal ensemble of components."

This module provides the actual content generation capabilities
that make "AIGC" (AI Generated Content) possible.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, Callable
from enum import Enum
import json
import os


class ToolCategory(Enum):
    """Categories of tools in the atomic library."""
    TEXT = "text"           # Text generation (LLM)
    IMAGE = "image"         # Image generation
    VIDEO = "video"         # Video generation
    AUDIO = "audio"         # Audio generation
    SEARCH = "search"       # Information retrieval
    TRANSFORM = "transform" # Content transformation
    ANALYSIS = "analysis"   # Content analysis
    UTILITY = "utility"     # Utility functions


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "metadata": self.metadata,
            "error": self.error
        }


@dataclass
class ToolSpec:
    """Specification for a tool."""
    name: str
    description: str
    category: ToolCategory
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the atomic library.
    
    Tools are the "GC" (Generated Content) part of AIGC.
    They produce actual outputs, not simulations.
    """
    
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute the tool with given inputs.
        
        Args:
            inputs: Tool-specific input parameters
            context: Optional execution context (workflow state, etc.)
        
        Returns:
            ToolResult with output or error
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against schema."""
        required = self.spec.input_schema.get("required", [])
        for req in required:
            if req not in inputs:
                return False
        return True


class LLMTool(BaseTool):
    """
    Text generation tool using LLM.
    
    The foundational AIGC capability - generates text content
    based on prompts and context.
    
    Supports OpenAI-compatible endpoints (z.ai, Together, etc.) via base_url.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider = provider
        self.model = model or self._default_model()
        self.api_key = api_key or os.getenv(self._api_key_env())
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None
    
    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514"
        }
        return defaults.get(self.provider, "gpt-4o")
    
    def _api_key_env(self) -> str:
        envs = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        return envs.get(self.provider, "OPENAI_API_KEY")
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="llm_generate",
            description="Generate text content using a large language model",
            category=ToolCategory.TEXT,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "The generation prompt"},
                    "system": {"type": "string", "description": "System prompt for context"},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"},
                    "temperature": {"type": "number", "description": "Sampling temperature"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Generated text"},
                    "tokens_used": {"type": "integer"}
                }
            },
            examples=[
                {
                    "input": {"prompt": "Write a haiku about coding"},
                    "output": {"text": "Lines of logic flow\nBugs hide in the whitespace deep\nCoffee saves the day"}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate text using the configured LLM."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: prompt"
            )
        
        prompt = inputs["prompt"]
        system = inputs.get("system", "You are a helpful creative assistant.")
        max_tokens = inputs.get("max_tokens", 2000)
        temperature = inputs.get("temperature", 0.7)
        
        # Add context to prompt if available
        if context:
            workflow_context = context.get("workflow_context", "")
            if workflow_context:
                prompt = f"{workflow_context}\n\n{prompt}"
        
        try:
            if self.provider == "openai":
                return await self._execute_openai(prompt, system, max_tokens, temperature)
            elif self.provider == "anthropic":
                return await self._execute_anthropic(prompt, system, max_tokens, temperature)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unsupported provider: {self.provider}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_openai(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float
    ) -> ToolResult:
        """Execute using OpenAI API (or compatible endpoint like z.ai)."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="openai package not installed. Run: pip install openai"
            )
        
        # Support OpenAI-compatible endpoints (z.ai, Together, etc.)
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        return ToolResult(
            success=True,
            output={"text": text, "tokens_used": tokens},
            metadata={"model": self.model, "provider": "openai"}
        )
    
    async def _execute_anthropic(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float
    ) -> ToolResult:
        """Execute using Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="anthropic package not installed. Run: pip install anthropic"
            )
        
        client = AsyncAnthropic(api_key=self.api_key)
        
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return ToolResult(
            success=True,
            output={"text": text, "tokens_used": tokens},
            metadata={"model": self.model, "provider": "anthropic"}
        )


class TemplateTool(BaseTool):
    """
    Template-based content generation tool.
    
    Useful for structured content where format is known
    but values need to be filled in.
    """
    
    def __init__(self):
        self._templates: Dict[str, str] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in templates."""
        self._templates["blog_post"] = """# {title}

{introduction}

## Key Points

{key_points}

## Conclusion

{conclusion}

---
*{footer}*
"""
        
        self._templates["social_post"] = """{hook}

{body}

{call_to_action}

{hashtags}"""
        
        self._templates["product_description"] = """## {product_name}

{tagline}

### Features
{features}

### Benefits
{benefits}

**{price_cta}**
"""
    
    def register_template(self, name: str, template: str) -> None:
        """Register a custom template."""
        self._templates[name] = template
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="template_fill",
            description="Fill a content template with provided values",
            category=ToolCategory.TEXT,
            input_schema={
                "type": "object",
                "required": ["template_name", "values"],
                "properties": {
                    "template_name": {"type": "string"},
                    "values": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Fill template with values."""
        template_name = inputs.get("template_name")
        values = inputs.get("values", {})
        
        if template_name not in self._templates:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown template: {template_name}. Available: {list(self._templates.keys())}"
            )
        
        template = self._templates[template_name]
        
        try:
            # Simple string formatting
            filled = template.format(**values)
            return ToolResult(
                success=True,
                output={"text": filled},
                metadata={"template": template_name}
            )
        except KeyError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Missing template value: {e}"
            )


class CombineTool(BaseTool):
    """
    Tool for combining multiple content pieces.
    
    Used when workflow has parallel branches that need
    to be merged into final output.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="combine",
            description="Combine multiple content pieces into one",
            category=ToolCategory.TRANSFORM,
            input_schema={
                "type": "object",
                "required": ["pieces"],
                "properties": {
                    "pieces": {"type": "array", "items": {"type": "string"}},
                    "separator": {"type": "string", "default": "\n\n"},
                    "order": {"type": "array", "items": {"type": "integer"}}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Combine content pieces."""
        pieces = inputs.get("pieces", [])
        separator = inputs.get("separator", "\n\n")
        order = inputs.get("order")
        
        if order:
            pieces = [pieces[i] for i in order if i < len(pieces)]
        
        combined = separator.join(str(p) for p in pieces if p)
        
        return ToolResult(
            success=True,
            output={"text": combined},
            metadata={"piece_count": len(pieces)}
        )


class ToolRegistry:
    """
    Registry for discovering and accessing tools.
    
    The MetaPlanner queries this registry to find
    appropriate tools for workflow nodes.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        name = tool.spec.name
        self._tools[name] = tool
        self._categories[tool.spec.category].append(name)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolSpec]:
        """List all tools, optionally filtered by category."""
        if category:
            names = self._categories.get(category, [])
            return [self._tools[n].spec for n in names]
        return [t.spec for t in self._tools.values()]
    
    def find_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Find all tools in a category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names]
    
    def to_prompt_context(self) -> str:
        """Generate prompt context listing available tools."""
        lines = ["## Available Tools\n"]
        
        for category in ToolCategory:
            tools = self.find_by_category(category)
            if tools:
                lines.append(f"\n### {category.value.title()} Tools:")
                for tool in tools:
                    lines.append(f"- **{tool.spec.name}**: {tool.spec.description}")
        
        return "\n".join(lines)


def create_default_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()
    
    # Register built-in tools
    registry.register(LLMTool(provider="openai"))
    registry.register(TemplateTool())
    registry.register(CombineTool())
    
    return registry
