# Changelog

All notable changes to Vibe AIGC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-02-05

### Added

- **General, Constraint-Aware Architecture** (Paper Section 5.4)
  - `discovery.py` - ComfyPilot-based system discovery (GPU, VRAM, nodes, models)
  - `composer_general.py` - Composes workflows from DISCOVERED nodes
  - `vibe_backend.py` - Unified backend with VLM feedback loop
  - `workflow_registry.py` - Workflows as first-class tools (select or compose)
  - `workflow_backend.py` - Unified workflow execution

- **True Generality**
  - No hardcoded model patterns — everything discovered from ComfyUI
  - Constraint-aware matching — filters by user's VRAM
  - CivitAI/HuggingFace API search — recommend models within constraints
  - Works on ANY ComfyUI setup (4GB laptop to 24GB workstation)

- **Workflow Templates**
  - `workflows/` directory for saved workflow JSON files
  - Workflows treated as atomic tools per paper Section 5.4
  - Auto-parameterization (prompt, seed, resolution)

### Changed

- `ModelRegistry` now detects Wan 2.x video models
- `MVPipeline` wired to new workflow backend
- Refactored from hardcoded patterns to discovery-based architecture

### Paper Alignment

- "Traverses atomic tool library" → `discovery.py`
- "Select optimal ensemble of components" → `workflow_registry.py`
- "Define data-flow topology" → `composer_general.py`
- "Adaptive reasoning for task complexity" → constraint-aware matching

## [0.2.0] - 2026-02-05

### Added

- **Specialized Agent Framework** (Paper Section 4)
  - `BaseAgent` abstract class with role-based capabilities
  - `WriterAgent` - Text content generation
  - `ResearcherAgent` - Information gathering
  - `EditorAgent` - Content refinement
  - `DirectorAgent` - Workflow coordination
  - `DesignerAgent` - Visual asset planning
  - `ScreenwriterAgent` - Script/narrative creation
  - `ComposerAgent` - Audio/music creation
  - `AgentRegistry` for agent discovery and team creation

- **Multi-Modal Content Generation Tools**
  - `ImageGenerationTool` - DALL-E 3, Replicate (Flux, SDXL)
  - `VideoGenerationTool` - Replicate video models
  - `AudioGenerationTool` - MusicGen, sound effects
  - `TTSTool` - ElevenLabs and OpenAI TTS
  - `SearchTool` - Brave Search API
  - `ScrapeTool` - Web page extraction
  - `create_full_registry()` for all tools

- **Asset Bank for Consistency**
  - `AssetBank` class for shared asset storage
  - `Character` profiles with visual consistency
  - `StyleGuide` for visual/tonal consistency
  - `Artifact` storage for generated content
  - Persistence to disk
  - Context generation for prompts

## [0.1.2] - 2026-02-05

### Added

- **Domain-Specific Expert Knowledge Base** (Paper Section 5.3)
  - `KnowledgeBase` class for storing domain expertise
  - Built-in knowledge for film, writing, design, and music domains
  - Query interface for MetaPlanner intent understanding
  - Maps creative concepts (e.g., "Hitchcockian suspense") to technical specs
  - `to_prompt_context()` for LLM integration

- **Atomic Tool Library** (Paper Section 5.4)
  - `ToolRegistry` for discovering and managing tools
  - `BaseTool` abstract class for custom tool implementations
  - `LLMTool` - Text generation using OpenAI or Anthropic
  - `TemplateTool` - Template-based content generation
  - `CombineTool` - Merge outputs from parallel workflow branches

- **Full Paper Architecture Integration**
  - MetaPlanner now uses KnowledgeBase for intent understanding
  - WorkflowExecutor integrates with ToolRegistry for real content generation
  - Nodes can specify which tool to use via parameters
  - Context flows between dependent nodes

### Changed

- MetaPlanner constructor accepts `knowledge_base` and `tool_registry` parameters
- WorkflowExecutor uses tools when available, falls back to simulation
- LLMClient.decompose_vibe accepts knowledge and tools context

## [0.1.1] - 2026-02-05

### Added

- **CLI Tool** - `vibe-aigc` command with plan/execute/checkpoints/resume subcommands
- **Docker Support** - Dockerfile for containerized deployments
- **Documentation Site** - Full MkDocs site with guides and API reference
- **Landing Page** - Custom homepage at jmanhype.github.io/vibe-aigc
- **Integration Examples** - OpenAI integration and custom executor examples
- **Security Policy** - SECURITY.md for vulnerability reporting
- **Code of Conduct** - CODE_OF_CONDUCT.md for community guidelines
- **Project Logo** - SVG logo for branding

### Changed

- CI now includes test coverage reporting (Codecov)
- README updated with CLI usage and coverage badge

## [0.1.0] - 2026-02-05

### Added

- **Core Architecture**
  - `Vibe` model for high-level intent representation
  - `WorkflowPlan` and `WorkflowNode` models for hierarchical task decomposition
  - `WorkflowNodeType` enum (ANALYZE, GENERATE, TRANSFORM, VALIDATE, COMPOSITE)

- **MetaPlanner**
  - LLM-based Vibe decomposition into executable workflows
  - `plan()` method for workflow generation
  - `execute()` method for plan-and-execute workflow
  - `execute_with_adaptation()` for automatic replanning on failures
  - `execute_with_visualization()` for real-time progress display
  - `execute_with_resume()` for checkpoint-based resumption

- **Execution Engine**
  - `WorkflowExecutor` with parallel execution support
  - Dependency-aware node scheduling
  - Progress callbacks for real-time monitoring
  - Execution status tracking (PENDING, RUNNING, COMPLETED, FAILED, SKIPPED)

- **Persistence**
  - `WorkflowPersistenceManager` for checkpoint management
  - Automatic checkpointing at configurable intervals
  - Checkpoint listing, retrieval, and deletion
  - Resume from any saved checkpoint

- **Visualization**
  - ASCII diagram generation for terminal output
  - Mermaid diagram generation for documentation
  - Status indicators for execution progress
  - Execution summary with timing information

- **LLM Integration**
  - OpenAI-compatible client
  - Configurable model, temperature, and token limits
  - Custom endpoint support

### Technical

- Python 3.12+ required
- Pydantic 2.0+ for data validation
- Full async/await support
- 128 passing tests
- MIT License

[0.1.0]: https://github.com/jmanhype/vibe-aigc/releases/tag/v0.1.0
