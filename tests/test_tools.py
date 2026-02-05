"""Tests for Atomic Tool Library (Paper Section 5.4)."""

import pytest
from vibe_aigc.tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ToolSpec,
    ToolCategory,
    LLMTool,
    TemplateTool,
    CombineTool,
    create_default_registry
)


class TestToolRegistry:
    """Test the tool registry functionality."""
    
    def test_create_default_registry(self):
        """Test that default registry has built-in tools."""
        registry = create_default_registry()
        
        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        
        assert "llm_generate" in tool_names
        assert "template_fill" in tool_names
        assert "combine" in tool_names
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = create_default_registry()
        
        llm = registry.get("llm_generate")
        
        assert llm is not None
        assert llm.spec.name == "llm_generate"
        assert llm.spec.category == ToolCategory.TEXT
    
    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = create_default_registry()
        
        tool = registry.get("nonexistent_tool")
        
        assert tool is None
    
    def test_find_by_category(self):
        """Test finding tools by category."""
        registry = create_default_registry()
        
        text_tools = registry.find_by_category(ToolCategory.TEXT)
        
        assert len(text_tools) >= 2  # llm_generate, template_fill
        assert all(t.spec.category == ToolCategory.TEXT for t in text_tools)
    
    def test_to_prompt_context(self):
        """Test generating prompt context for tools."""
        registry = create_default_registry()
        
        context = registry.to_prompt_context()
        
        assert "Available Tools" in context
        assert "llm_generate" in context
        assert "template_fill" in context


class TestToolResult:
    """Test ToolResult class."""
    
    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolResult(
            success=True,
            output={"text": "Hello world"},
            metadata={"model": "test"}
        )
        
        assert result.success is True
        assert result.output["text"] == "Hello world"
        assert result.error is None
    
    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            success=False,
            output=None,
            error="Something went wrong"
        )
        
        assert result.success is False
        assert result.output is None
        assert result.error == "Something went wrong"
    
    def test_to_dict(self):
        """Test converting result to dict."""
        result = ToolResult(
            success=True,
            output="test",
            metadata={"key": "value"}
        )
        
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["output"] == "test"
        assert d["metadata"]["key"] == "value"


class TestTemplateTool:
    """Test TemplateTool functionality."""
    
    @pytest.mark.asyncio
    async def test_fill_blog_post_template(self):
        """Test filling a blog post template."""
        tool = TemplateTool()
        
        result = await tool.execute({
            "template_name": "blog_post",
            "values": {
                "title": "Test Post",
                "introduction": "This is the intro.",
                "key_points": "- Point 1\n- Point 2",
                "conclusion": "In conclusion...",
                "footer": "Written by AI"
            }
        })
        
        assert result.success is True
        assert "Test Post" in result.output["text"]
        assert "This is the intro" in result.output["text"]
        assert "Point 1" in result.output["text"]
    
    @pytest.mark.asyncio
    async def test_fill_social_post_template(self):
        """Test filling a social post template."""
        tool = TemplateTool()
        
        result = await tool.execute({
            "template_name": "social_post",
            "values": {
                "hook": "Did you know?",
                "body": "Interesting fact here",
                "call_to_action": "Share this!",
                "hashtags": "#ai #test"
            }
        })
        
        assert result.success is True
        assert "Did you know?" in result.output["text"]
        assert "#ai #test" in result.output["text"]
    
    @pytest.mark.asyncio
    async def test_unknown_template(self):
        """Test filling an unknown template."""
        tool = TemplateTool()
        
        result = await tool.execute({
            "template_name": "nonexistent",
            "values": {}
        })
        
        assert result.success is False
        assert "Unknown template" in result.error
    
    @pytest.mark.asyncio
    async def test_missing_template_value(self):
        """Test filling template with missing values."""
        tool = TemplateTool()
        
        result = await tool.execute({
            "template_name": "blog_post",
            "values": {
                "title": "Test"
                # Missing other required values
            }
        })
        
        assert result.success is False
        assert "Missing template value" in result.error
    
    def test_register_custom_template(self):
        """Test registering a custom template."""
        tool = TemplateTool()
        
        tool.register_template("custom", "Hello {name}!")
        
        # Verify it's registered (would need to execute to fully test)
        assert "custom" in tool._templates


class TestCombineTool:
    """Test CombineTool functionality."""
    
    @pytest.mark.asyncio
    async def test_combine_pieces(self):
        """Test combining content pieces."""
        tool = CombineTool()
        
        result = await tool.execute({
            "pieces": ["Part 1", "Part 2", "Part 3"],
            "separator": "\n---\n"
        })
        
        assert result.success is True
        assert "Part 1" in result.output["text"]
        assert "Part 2" in result.output["text"]
        assert "---" in result.output["text"]
    
    @pytest.mark.asyncio
    async def test_combine_with_order(self):
        """Test combining with custom order."""
        tool = CombineTool()
        
        result = await tool.execute({
            "pieces": ["A", "B", "C"],
            "order": [2, 0, 1]  # C, A, B
        })
        
        assert result.success is True
        text = result.output["text"]
        # C should come before A, A before B
        assert text.index("C") < text.index("A")
        assert text.index("A") < text.index("B")
    
    @pytest.mark.asyncio
    async def test_combine_empty(self):
        """Test combining empty pieces."""
        tool = CombineTool()
        
        result = await tool.execute({
            "pieces": []
        })
        
        assert result.success is True
        assert result.output["text"] == ""


class TestLLMTool:
    """Test LLMTool functionality (without actual API calls)."""
    
    def test_spec(self):
        """Test LLMTool specification."""
        tool = LLMTool()
        
        spec = tool.spec
        
        assert spec.name == "llm_generate"
        assert spec.category == ToolCategory.TEXT
        assert "prompt" in spec.input_schema["required"]
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        tool = LLMTool()
        
        assert tool.validate_inputs({"prompt": "Hello"}) is True
    
    def test_validate_inputs_invalid(self):
        """Test input validation with missing required input."""
        tool = LLMTool()
        
        assert tool.validate_inputs({}) is False
        assert tool.validate_inputs({"system": "test"}) is False
    
    @pytest.mark.asyncio
    async def test_execute_without_prompt(self):
        """Test execution fails without prompt."""
        tool = LLMTool()
        
        result = await tool.execute({})
        
        assert result.success is False
        assert "Missing required input" in result.error
