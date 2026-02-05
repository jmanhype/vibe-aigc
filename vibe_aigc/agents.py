"""
Specialized Agent Framework.

Based on Paper Section 4 examples:
- AutoPR: Logical Draft, Visual Analysis, Textual Enriching agents
- AutoMV: Screenwriter Agent, Director Agent
- Poster Copilot: Layout reasoning agents

Agents are role-based entities that use tools to accomplish tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import asyncio

if TYPE_CHECKING:
    from .tools import ToolRegistry, ToolResult


class AgentRole(Enum):
    """Predefined agent roles based on paper examples."""
    WRITER = "writer"           # Text content generation
    RESEARCHER = "researcher"   # Information gathering
    EDITOR = "editor"           # Content refinement
    DIRECTOR = "director"       # Workflow coordination
    DESIGNER = "designer"       # Visual asset planning
    SCREENWRITER = "screenwriter"  # Script/narrative creation
    ANALYST = "analyst"         # Data/content analysis
    COMPOSER = "composer"       # Audio/music creation
    ANIMATOR = "animator"       # Video/motion creation


@dataclass
class AgentContext:
    """Context passed to agents during execution."""
    task: str
    vibe_description: str
    style: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    previous_outputs: Dict[str, Any] = field(default_factory=dict)
    shared_assets: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    output: Any
    artifacts: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "artifacts": self.artifacts,
            "messages": self.messages,
            "metadata": self.metadata
        }


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    
    Agents combine:
    - A role (what they do)
    - Capabilities (tools they can use)
    - Personality (how they approach tasks)
    """
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        tool_registry: Optional['ToolRegistry'] = None,
        personality: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.tool_registry = tool_registry
        self.personality = personality or self._default_personality()
        self._capabilities: List[str] = []
    
    @abstractmethod
    def _default_personality(self) -> str:
        """Return default personality/system prompt for this agent type."""
        pass
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task given context."""
        pass
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent has capability to use a tool."""
        return tool_name in self._capabilities or not self._capabilities
    
    async def use_tool(self, tool_name: str, inputs: Dict[str, Any]) -> 'ToolResult':
        """Use a tool from the registry."""
        if not self.tool_registry:
            raise RuntimeError(f"Agent {self.name} has no tool registry")
        
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        if not self.can_use_tool(tool_name):
            raise PermissionError(f"Agent {self.name} cannot use tool: {tool_name}")
        
        return await tool.execute(inputs)


class WriterAgent(BaseAgent):
    """Agent specialized in text content generation."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Writer"):
        super().__init__(name, AgentRole.WRITER, tool_registry)
        self._capabilities = ["llm_generate", "template_fill", "combine"]
    
    def _default_personality(self) -> str:
        return """You are a skilled content writer. You create engaging, clear, 
and well-structured text content. You adapt your tone and style to match 
the project's vibe while maintaining quality and readability."""
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Generate text content based on context."""
        if not self.tool_registry:
            return AgentResult(
                success=False,
                output=None,
                messages=["No tool registry available"]
            )
        
        # Build prompt from context
        prompt = f"""Task: {context.task}

Project Vibe: {context.vibe_description}
Style: {context.style or 'Not specified'}
Constraints: {', '.join(context.constraints) if context.constraints else 'None'}

{self._format_previous_outputs(context.previous_outputs)}

Generate the requested content."""

        result = await self.use_tool("llm_generate", {
            "prompt": prompt,
            "system": self.personality
        })
        
        if result.success:
            return AgentResult(
                success=True,
                output=result.output.get("text", result.output),
                metadata={"tool": "llm_generate", "tokens": result.output.get("tokens_used")}
            )
        else:
            return AgentResult(
                success=False,
                output=None,
                messages=[result.error or "Unknown error"]
            )
    
    def _format_previous_outputs(self, outputs: Dict[str, Any]) -> str:
        if not outputs:
            return ""
        lines = ["Previous work:"]
        for key, value in outputs.items():
            if isinstance(value, dict) and "text" in value:
                lines.append(f"- {key}: {value['text'][:200]}...")
            else:
                lines.append(f"- {key}: {str(value)[:200]}...")
        return "\n".join(lines)


class ResearcherAgent(BaseAgent):
    """Agent specialized in information gathering and research."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Researcher"):
        super().__init__(name, AgentRole.RESEARCHER, tool_registry)
        self._capabilities = ["search", "scrape", "llm_generate"]
    
    def _default_personality(self) -> str:
        return """You are a thorough researcher. You gather accurate information 
from multiple sources, verify facts, and synthesize findings into clear summaries.
You cite sources and distinguish between facts and opinions."""
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Research a topic and return findings."""
        # Try search tool if available
        search_result = None
        if self.tool_registry and self.tool_registry.get("search"):
            search_result = await self.use_tool("search", {
                "query": context.task
            })
        
        # Synthesize with LLM
        prompt = f"""Research Task: {context.task}

Project Context: {context.vibe_description}

{f"Search Results: {search_result.output}" if search_result and search_result.success else ""}

Provide a comprehensive research summary with key findings."""

        if self.tool_registry and self.tool_registry.get("llm_generate"):
            result = await self.use_tool("llm_generate", {
                "prompt": prompt,
                "system": self.personality
            })
            
            if result.success:
                return AgentResult(
                    success=True,
                    output=result.output.get("text", result.output),
                    artifacts={"search_results": search_result.output if search_result else None},
                    metadata={"tool": "llm_generate"}
                )
        
        return AgentResult(
            success=False,
            output=None,
            messages=["Could not complete research - missing tools"]
        )


class EditorAgent(BaseAgent):
    """Agent specialized in content refinement and quality improvement."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Editor"):
        super().__init__(name, AgentRole.EDITOR, tool_registry)
        self._capabilities = ["llm_generate", "combine"]
    
    def _default_personality(self) -> str:
        return """You are a meticulous editor. You improve clarity, fix errors,
enhance flow, and ensure content matches the intended style and tone.
You preserve the author's voice while elevating quality."""
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Edit and refine content."""
        # Get content to edit from previous outputs
        content_to_edit = None
        for key, value in context.previous_outputs.items():
            if isinstance(value, dict) and "text" in value:
                content_to_edit = value["text"]
                break
            elif isinstance(value, str):
                content_to_edit = value
                break
        
        if not content_to_edit:
            return AgentResult(
                success=False,
                output=None,
                messages=["No content provided to edit"]
            )
        
        prompt = f"""Edit Task: {context.task}

Style Guidelines: {context.style or 'Match original tone'}
Constraints: {', '.join(context.constraints) if context.constraints else 'None'}

Content to Edit:
{content_to_edit}

Provide the edited version with improvements."""

        if self.tool_registry and self.tool_registry.get("llm_generate"):
            result = await self.use_tool("llm_generate", {
                "prompt": prompt,
                "system": self.personality
            })
            
            if result.success:
                return AgentResult(
                    success=True,
                    output=result.output.get("text", result.output),
                    metadata={"tool": "llm_generate", "original_length": len(content_to_edit)}
                )
        
        return AgentResult(
            success=False,
            output=None,
            messages=["Could not complete editing"]
        )


class DirectorAgent(BaseAgent):
    """Agent specialized in coordinating workflows and other agents."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Director"):
        super().__init__(name, AgentRole.DIRECTOR, tool_registry)
        self._capabilities = ["llm_generate"]
        self._managed_agents: Dict[str, BaseAgent] = {}
    
    def _default_personality(self) -> str:
        return """You are a creative director. You coordinate complex projects,
delegate tasks to specialists, maintain consistency across deliverables,
and ensure the final output matches the creative vision."""
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to manage."""
        self._managed_agents[agent.name] = agent
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Coordinate a complex task using managed agents."""
        # Plan the work
        plan_prompt = f"""Project: {context.vibe_description}
Current Task: {context.task}
Style: {context.style}
Constraints: {', '.join(context.constraints) if context.constraints else 'None'}

Available team members: {list(self._managed_agents.keys())}

Create a brief action plan: who should do what, in what order."""

        if self.tool_registry and self.tool_registry.get("llm_generate"):
            plan_result = await self.use_tool("llm_generate", {
                "prompt": plan_prompt,
                "system": self.personality
            })
            
            if plan_result.success:
                return AgentResult(
                    success=True,
                    output=plan_result.output.get("text", plan_result.output),
                    metadata={
                        "role": "coordination",
                        "managed_agents": list(self._managed_agents.keys())
                    }
                )
        
        return AgentResult(
            success=False,
            output=None,
            messages=["Could not create coordination plan"]
        )


class DesignerAgent(BaseAgent):
    """Agent specialized in visual design and asset creation."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Designer"):
        super().__init__(name, AgentRole.DESIGNER, tool_registry)
        self._capabilities = ["image_generate", "llm_generate"]
    
    def _default_personality(self) -> str:
        return """You are a visual designer. You create compelling visual concepts,
design assets, and ensure visual consistency across projects. You understand
color theory, composition, typography, and brand identity."""
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Create visual design or generate images."""
        # Check for image generation tool
        if self.tool_registry and self.tool_registry.get("image_generate"):
            result = await self.use_tool("image_generate", {
                "prompt": f"{context.task}. Style: {context.style or 'modern, clean'}",
                "size": context.metadata.get("image_size", "1024x1024")
            })
            
            if result.success:
                return AgentResult(
                    success=True,
                    output=result.output,
                    artifacts={"image": result.output},
                    metadata={"tool": "image_generate"}
                )
        
        # Fallback to design description
        if self.tool_registry and self.tool_registry.get("llm_generate"):
            prompt = f"""Design Task: {context.task}
Project Vibe: {context.vibe_description}
Style: {context.style}

Create a detailed visual design specification including:
- Color palette
- Typography recommendations
- Layout suggestions
- Visual elements and imagery"""

            result = await self.use_tool("llm_generate", {
                "prompt": prompt,
                "system": self.personality
            })
            
            if result.success:
                return AgentResult(
                    success=True,
                    output=result.output.get("text", result.output),
                    metadata={"tool": "llm_generate", "type": "design_spec"}
                )
        
        return AgentResult(
            success=False,
            output=None,
            messages=["Could not complete design task"]
        )


class ScreenwriterAgent(BaseAgent):
    """Agent specialized in script and narrative creation (AutoMV example)."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Screenwriter"):
        super().__init__(name, AgentRole.SCREENWRITER, tool_registry)
        self._capabilities = ["llm_generate"]
    
    def _default_personality(self) -> str:
        return """You are a screenwriter. You craft compelling narratives,
write dialogue, create scene descriptions, and structure stories for
visual media. You understand pacing, character development, and visual storytelling."""
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Write scripts, narratives, or scene descriptions."""
        prompt = f"""Screenwriting Task: {context.task}

Project: {context.vibe_description}
Style/Tone: {context.style or 'cinematic'}
Constraints: {', '.join(context.constraints) if context.constraints else 'None'}

{self._format_context(context)}

Write the requested script/narrative content with scene descriptions, dialogue, and visual notes."""

        if self.tool_registry and self.tool_registry.get("llm_generate"):
            result = await self.use_tool("llm_generate", {
                "prompt": prompt,
                "system": self.personality,
                "max_tokens": 3000
            })
            
            if result.success:
                return AgentResult(
                    success=True,
                    output=result.output.get("text", result.output),
                    metadata={"tool": "llm_generate", "type": "script"}
                )
        
        return AgentResult(
            success=False,
            output=None,
            messages=["Could not complete screenwriting task"]
        )
    
    def _format_context(self, context: AgentContext) -> str:
        parts = []
        if context.shared_assets.get("characters"):
            parts.append(f"Characters: {context.shared_assets['characters']}")
        if context.shared_assets.get("setting"):
            parts.append(f"Setting: {context.shared_assets['setting']}")
        if context.metadata.get("music_info"):
            parts.append(f"Music/Audio: {context.metadata['music_info']}")
        return "\n".join(parts)


class ComposerAgent(BaseAgent):
    """Agent specialized in audio and music creation."""
    
    def __init__(self, tool_registry: Optional['ToolRegistry'] = None, name: str = "Composer"):
        super().__init__(name, AgentRole.COMPOSER, tool_registry)
        self._capabilities = ["audio_generate", "tts", "llm_generate"]
    
    def _default_personality(self) -> str:
        return """You are a composer and audio designer. You create music,
sound effects, and audio content. You understand rhythm, melody, harmony,
and how audio enhances visual storytelling."""
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Generate audio content or music."""
        # Check for audio tools
        if self.tool_registry:
            # Try TTS for voiceover
            if context.metadata.get("type") == "voiceover" and self.tool_registry.get("tts"):
                text = context.previous_outputs.get("script", {}).get("text", context.task)
                result = await self.use_tool("tts", {
                    "text": text,
                    "voice": context.metadata.get("voice", "default")
                })
                if result.success:
                    return AgentResult(
                        success=True,
                        output=result.output,
                        artifacts={"audio": result.output},
                        metadata={"tool": "tts", "type": "voiceover"}
                    )
            
            # Try music generation
            if self.tool_registry.get("audio_generate"):
                result = await self.use_tool("audio_generate", {
                    "prompt": f"{context.task}. Style: {context.style}",
                    "duration": context.metadata.get("duration", 30)
                })
                if result.success:
                    return AgentResult(
                        success=True,
                        output=result.output,
                        artifacts={"audio": result.output},
                        metadata={"tool": "audio_generate"}
                    )
        
        # Fallback to music description
        if self.tool_registry and self.tool_registry.get("llm_generate"):
            prompt = f"""Audio/Music Task: {context.task}
Project: {context.vibe_description}
Style: {context.style}

Create a detailed audio design specification including:
- Tempo and rhythm
- Instrumentation
- Mood progression
- Key moments/transitions"""

            result = await self.use_tool("llm_generate", {
                "prompt": prompt,
                "system": self.personality
            })
            
            if result.success:
                return AgentResult(
                    success=True,
                    output=result.output.get("text", result.output),
                    metadata={"tool": "llm_generate", "type": "audio_spec"}
                )
        
        return AgentResult(
            success=False,
            output=None,
            messages=["Could not complete audio task"]
        )


class AgentRegistry:
    """Registry for discovering and managing agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._roles: Dict[AgentRole, List[str]] = {role: [] for role in AgentRole}
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent
        self._roles[agent.role].append(agent.name)
    
    def get(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agents.get(name)
    
    def find_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """Find all agents with a specific role."""
        return [self._agents[name] for name in self._roles.get(role, [])]
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    def create_team(self, roles: List[AgentRole]) -> Dict[AgentRole, BaseAgent]:
        """Create a team with one agent per role."""
        team = {}
        for role in roles:
            agents = self.find_by_role(role)
            if agents:
                team[role] = agents[0]
        return team


def create_default_agents(tool_registry: Optional['ToolRegistry'] = None) -> AgentRegistry:
    """Create a registry with default agents."""
    registry = AgentRegistry()
    
    registry.register(WriterAgent(tool_registry))
    registry.register(ResearcherAgent(tool_registry))
    registry.register(EditorAgent(tool_registry))
    registry.register(DirectorAgent(tool_registry))
    registry.register(DesignerAgent(tool_registry))
    registry.register(ScreenwriterAgent(tool_registry))
    registry.register(ComposerAgent(tool_registry))
    
    return registry
