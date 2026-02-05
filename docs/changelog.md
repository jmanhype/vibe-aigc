# Changelog

All notable changes to Vibe AIGC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
