# Research: Core Vibe Model and Meta-Planner Architecture

**Date**: February 5, 2026
**Item**: 001-vibe-model-meta-planner

## Research Question

Implement the foundational Vibe AIGC system based on arXiv:2602.04575. Create the Vibe data model (high-level aesthetic/intent representation) and MetaPlanner class that decomposes Vibes into executable agentic pipelines.

**Motivation:** Bridge the Intent-Execution Gap by enabling users to express high-level creative intent (Vibe) and have the system automatically generate and execute hierarchical agent workflows.

**Success criteria:**
- Vibe dataclass with description, style, constraints, domain fields
- MetaPlanner class that accepts Vibe and generates WorkflowPlan
- WorkflowNode structure for hierarchical task decomposition
- Integration with LLM for intelligent decomposition
- Basic execution engine for running generated workflows

**Technical constraints:**
- Python 3.12+
- Pydantic for data validation
- Async/await for execution
- OpenAI-compatible LLM client

## Summary

This is a greenfield implementation where we need to build the foundational components of the Vibe AIGC system from scratch. The project currently contains only documentation (README.md) and project management infrastructure (.wreckit). No source code exists yet, giving us complete freedom to design the architecture according to the paper's specifications and modern Python best practices.

The core challenge is implementing a system that can take high-level creative intent ("Vibe") and automatically decompose it into executable agent workflows. This requires creating robust data models, an intelligent planning system that leverages LLMs for decomposition, and an execution engine that can run hierarchical multi-agent workflows asynchronously.

Based on the README.md usage example, the intended API surface is clean and intuitive, suggesting we need to build a well-architected system that hides complexity behind simple interfaces while maintaining flexibility for complex workflows.

## Current State Analysis

### Existing Implementation

- **No source code exists** - This is a completely greenfield project
- `README.md:1-56` - Contains project overview, conceptual architecture, and intended usage example
- `README.md:39-50` - Shows expected API: `from vibe_aigc import MetaPlanner, Vibe`
- `README.md:25` - Defines architecture flow: `User Vibe → Meta-Planner → Agentic Pipeline → Execution → Result`
- `.wreckit/items/001-vibe-model-meta-planner/item.json:1-29` - Contains detailed requirements and technical constraints
- `.gitignore:1-4` - Minimal gitignore, only excludes wreckit local config

### Project Structure Expectations

Based on the import statement in README.md:39, we need to create a `vibe_aigc` Python package with the following modules:
- `Vibe` dataclass - High-level intent representation
- `MetaPlanner` class - Core orchestration and decomposition logic

## Key Files

- `README.md:9` - Defines Vibe as "high-level representation encompassing aesthetic preferences, functional logic, and intent"
- `README.md:17` - Describes MetaPlanner as "centralized system architect that deconstructs a user's 'Vibe' into executable, verifiable, and adaptive agentic pipelines"
- `README.md:42-46` - Shows expected Vibe constructor with description, style, constraints fields
- `README.md:49-50` - Shows MetaPlanner.execute() method as the primary interface
- `.wreckit/items/001-vibe-model-meta-planner/item.json:17-21` - Lists specific success criteria including WorkflowPlan and WorkflowNode structures

## Technical Considerations

### Dependencies

**External dependencies needed:**
- `pydantic>=2.0` - For data validation and model definition (item.json:25)
- `openai` or similar OpenAI-compatible client - For LLM integration (item.json:27)
- `asyncio` built-in - For async/await execution patterns (item.json:26)
- Python 3.12+ as base requirement (item.json:24)

**Internal modules to create:**
- `vibe_aigc/models.py` - Vibe, WorkflowPlan, WorkflowNode dataclasses
- `vibe_aigc/planner.py` - MetaPlanner class with LLM integration
- `vibe_aigc/executor.py` - Basic execution engine for running workflows
- `vibe_aigc/__init__.py` - Package exports matching README.md:39

### Patterns to Follow

**Architecture patterns from README.md:**
- Clean separation between Intent (Vibe) and Execution (WorkflowPlan)
- Hierarchical decomposition through WorkflowNode structures
- Async/await for all I/O operations including LLM calls
- Feedback loops for adaptive planning (README.md:27)

**API Design principles:**
- Simple constructor-based API for Vibe creation
- Single execute() method as primary MetaPlanner interface
- Immutable data structures using Pydantic models

## Risks and Mitigations

| Risk | Impact | Mitigation |
| -------- | ----------------- | ---------------- |
| No existing patterns to follow | Medium | Research similar agent orchestration systems, use established Python patterns for async workflows |
| LLM integration complexity | High | Start with simple OpenAI client wrapper, design for pluggable LLM backends |
| Hierarchical workflow execution | High | Begin with simple linear execution, gradually add hierarchy support |
| Vibe representation ambiguity | Medium | Start with explicit fields (description, style, constraints), extend based on usage |
| Paper implementation details missing | High | Focus on core concepts from README.md, iterate based on practical usage |

## Recommended Approach

**Phase 1: Core Data Models**
Create the foundational Pydantic models (Vibe, WorkflowPlan, WorkflowNode) with clear field definitions matching the README.md usage example. This establishes the API contract and enables early testing.

**Phase 2: Basic MetaPlanner**
Implement a simple MetaPlanner that can accept a Vibe and generate a minimal WorkflowPlan using LLM integration. Focus on the decomposition logic that transforms high-level intent into concrete tasks.

**Phase 3: Execution Engine**
Build a basic async execution engine that can run generated WorkflowPlans. Start with linear execution before adding hierarchical capabilities.

**Phase 4: Integration and Testing**
Wire everything together to match the README.md usage example, add comprehensive testing, and validate the end-to-end workflow.

This approach prioritizes getting a working system quickly while leaving room for sophisticated features like adaptive planning and complex hierarchical workflows.

## Open Questions

- **LLM Provider**: While "OpenAI-compatible" is specified, should we default to OpenAI's client or create an abstraction layer?
- **WorkflowNode Structure**: How deep should the hierarchy go? What are the atomic operations?
- **Execution State Management**: How should we handle partial failures, retries, and workflow state persistence?
- **Vibe Extensions**: Beyond description/style/constraints, what other fields might be needed for domain-specific use cases?
- **Feedback Mechanism**: The README.md:27 shows a feedback loop - how should this be implemented in practice?