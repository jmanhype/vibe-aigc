"""Tests for Specialized Agent Framework."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from vibe_aigc.agents import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentResult,
    AgentRegistry,
    WriterAgent,
    ResearcherAgent,
    EditorAgent,
    DirectorAgent,
    DesignerAgent,
    ScreenwriterAgent,
    ComposerAgent,
    create_default_agents
)
from vibe_aigc.tools import ToolRegistry, ToolResult


class TestAgentContext:
    """Test AgentContext class."""
    
    def test_create_context(self):
        ctx = AgentContext(
            task="Write a blog post",
            vibe_description="Technical but accessible",
            style="informative",
            constraints=["under 1000 words"]
        )
        
        assert ctx.task == "Write a blog post"
        assert ctx.style == "informative"
        assert len(ctx.constraints) == 1


class TestAgentResult:
    """Test AgentResult class."""
    
    def test_success_result(self):
        result = AgentResult(
            success=True,
            output="Generated content",
            artifacts={"image": "url"},
            messages=["Completed"]
        )
        
        assert result.success
        assert result.output == "Generated content"
        assert "image" in result.artifacts
    
    def test_to_dict(self):
        result = AgentResult(success=True, output="test")
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["output"] == "test"


class TestWriterAgent:
    """Test WriterAgent."""
    
    def test_creation(self):
        agent = WriterAgent()
        
        assert agent.name == "Writer"
        assert agent.role == AgentRole.WRITER
        assert "llm_generate" in agent._capabilities
    
    @pytest.mark.asyncio
    async def test_execute_without_registry(self):
        agent = WriterAgent()
        ctx = AgentContext(task="Write something", vibe_description="Test")
        
        result = await agent.execute(ctx)
        
        assert not result.success
        assert "No tool registry" in result.messages[0]
    
    @pytest.mark.asyncio
    async def test_execute_with_mock_registry(self):
        # Create mock registry
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_tool = AsyncMock()
        mock_tool.execute.return_value = ToolResult(
            success=True,
            output={"text": "Generated text", "tokens_used": 100}
        )
        mock_registry.get.return_value = mock_tool
        
        agent = WriterAgent(tool_registry=mock_registry)
        ctx = AgentContext(task="Write a haiku", vibe_description="Peaceful")
        
        result = await agent.execute(ctx)
        
        assert result.success
        assert "Generated text" in result.output


class TestResearcherAgent:
    """Test ResearcherAgent."""
    
    def test_creation(self):
        agent = ResearcherAgent()
        
        assert agent.role == AgentRole.RESEARCHER
        assert "search" in agent._capabilities


class TestEditorAgent:
    """Test EditorAgent."""
    
    def test_creation(self):
        agent = EditorAgent()
        
        assert agent.role == AgentRole.EDITOR
    
    @pytest.mark.asyncio
    async def test_execute_without_content(self):
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get.return_value = None
        
        agent = EditorAgent(tool_registry=mock_registry)
        ctx = AgentContext(task="Edit", vibe_description="Test")
        
        result = await agent.execute(ctx)
        
        assert not result.success
        assert "No content" in result.messages[0]


class TestDirectorAgent:
    """Test DirectorAgent."""
    
    def test_add_managed_agent(self):
        director = DirectorAgent()
        writer = WriterAgent()
        
        director.add_agent(writer)
        
        assert "Writer" in director._managed_agents


class TestDesignerAgent:
    """Test DesignerAgent."""
    
    def test_creation(self):
        agent = DesignerAgent()
        
        assert agent.role == AgentRole.DESIGNER
        assert "image_generate" in agent._capabilities


class TestScreenwriterAgent:
    """Test ScreenwriterAgent."""
    
    def test_creation(self):
        agent = ScreenwriterAgent()
        
        assert agent.role == AgentRole.SCREENWRITER


class TestComposerAgent:
    """Test ComposerAgent."""
    
    def test_creation(self):
        agent = ComposerAgent()
        
        assert agent.role == AgentRole.COMPOSER
        assert "audio_generate" in agent._capabilities


class TestAgentRegistry:
    """Test AgentRegistry."""
    
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = WriterAgent()
        
        registry.register(agent)
        
        assert registry.get("Writer") == agent
    
    def test_find_by_role(self):
        registry = AgentRegistry()
        registry.register(WriterAgent())
        registry.register(EditorAgent())
        
        writers = registry.find_by_role(AgentRole.WRITER)
        
        assert len(writers) == 1
        assert writers[0].role == AgentRole.WRITER
    
    def test_list_agents(self):
        registry = AgentRegistry()
        registry.register(WriterAgent())
        registry.register(EditorAgent())
        
        names = registry.list_agents()
        
        assert "Writer" in names
        assert "Editor" in names
    
    def test_create_team(self):
        registry = AgentRegistry()
        registry.register(WriterAgent())
        registry.register(EditorAgent())
        registry.register(DirectorAgent())
        
        team = registry.create_team([AgentRole.WRITER, AgentRole.EDITOR])
        
        assert AgentRole.WRITER in team
        assert AgentRole.EDITOR in team


class TestCreateDefaultAgents:
    """Test create_default_agents factory."""
    
    def test_creates_all_agents(self):
        registry = create_default_agents()
        
        assert registry.get("Writer") is not None
        assert registry.get("Researcher") is not None
        assert registry.get("Editor") is not None
        assert registry.get("Director") is not None
        assert registry.get("Designer") is not None
        assert registry.get("Screenwriter") is not None
        assert registry.get("Composer") is not None
