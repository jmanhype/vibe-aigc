# Research: Advanced Execution with Feedback Loops and Adaptive Planning

**Date**: February 5, 2026
**Item**: 002-advanced-execution-feedback

## Research Question

Implement the feedback loop mechanism from the Vibe AIGC paper (arXiv:2602.04575). Add adaptive planning that can refine workflows based on execution results, implement parallel execution for independent nodes, and add workflow visualization.

**Motivation:** The paper's architecture shows a feedback loop from execution back to the Meta-Planner. This enables adaptive workflows that can recover from failures and optimize based on results.

**Success criteria:**
- Feedback mechanism that sends execution results back to MetaPlanner
- Adaptive replanning when nodes fail
- Parallel execution of independent workflow nodes
- Workflow visualization (ASCII/Mermaid diagrams)
- Progress callbacks during execution
- Workflow persistence and resume capability

**Technical constraints:**
- Must integrate with existing MetaPlanner and WorkflowExecutor
- Use asyncio.gather for parallel execution
- Maintain backward compatibility with existing API

## Summary

The current Vibe AIGC system has all foundational components in place but executes workflows sequentially without feedback or adaptation. The architecture already supports the structures needed for advanced execution - hierarchical workflow nodes, dependency tracking, and async execution patterns. The main gaps are: 1) Sequential execution of independent nodes where parallel execution would be more efficient, 2) No feedback mechanism from execution results back to the MetaPlanner for adaptive replanning, and 3) Limited visibility into execution progress and state.

The existing codebase provides excellent integration points through the MetaPlanner-WorkflowExecutor relationship and the comprehensive error handling patterns. The challenge is extending the execution engine to support parallel execution while maintaining dependency order, implementing a feedback loop that allows the MetaPlanner to respond to execution results, and adding visualization capabilities without disrupting the clean API surface.

## Current State Analysis

### Existing Implementation

**Core Architecture (`vibe_aigc/__init__.py:3-6`):**
- Well-defined separation between models, planner, executor, and LLM client
- Comprehensive data models with Vibe, WorkflowPlan, WorkflowNode, and execution results
- Async execution patterns already established

**Sequential Execution Pattern (`vibe_aigc/executor.py:95-96`):**
- Root nodes executed sequentially: `for node in plan.root_nodes:`
- Children also executed sequentially (`executor.py:150-152`)
- No parallel execution despite independent nodes having no dependencies

**Dependency Management (`vibe_aigc/executor.py:162-169`):**
- Robust dependency checking in `_dependencies_satisfied()` method
- Supports node dependencies through `node.dependencies` list
- Foundation for parallel execution of independent nodes already exists

**Error Handling (`vibe_aigc/executor.py:99-102, 154-160`):**
- Comprehensive failure handling with tree marking
- Execution continues on failure but marks entire subtrees as failed
- No retry or adaptive replanning mechanism

**Integration Points:**
- `MetaPlanner.executor` relationship (`planner.py:17`) provides feedback pathway
- `ExecutionResult` class (`executor.py:34-72`) captures comprehensive execution data
- `WorkflowNode.dependencies` (`models.py:35`) already tracks execution order requirements

## Key Files

- `vibe_aigc/executor.py:95-96` - Sequential root node execution loop to parallelize
- `vibe_aigc/executor.py:150-152` - Sequential child execution to parallelize
- `vibe_aigc/executor.py:162-169` - Dependency resolution logic for parallel execution safety
- `vibe_aigc/models.py:35` - WorkflowNode.dependencies field for execution ordering
- `vibe_aigc/planner.py:97-102` - MetaPlanner.plan_and_execute() integration point for feedback
- `vibe_aigc/executor.py:34-72` - ExecutionResult class to extend for feedback data
- `tests/test_executor.py:81-120` - Dependency handling tests to extend for parallel execution
- `README.md:25-27` - Architecture diagram showing feedback loop to implement

## Technical Considerations

### Dependencies

**External dependencies needed:**
- No new external dependencies required - asyncio.gather is built-in
- Consider `graphviz` or `mermaid` libraries for visualization (optional)

**Internal modules to integrate with:**
- `vibe_aigc.executor.WorkflowExecutor` - Core execution engine to enhance
- `vibe_aigc.planner.MetaPlanner` - Feedback recipient for adaptive planning
- `vibe_aigc.models.ExecutionResult` - Data structure to extend for feedback
- `vibe_aigc.models.WorkflowNode` - Dependency model for parallel execution

### Patterns to Follow

**Async Execution Patterns (`tests/test_executor.py`, `vibe_aigc/executor.py`):**
- All execution methods use `async def` and `await`
- Error handling with try/catch and proper result tracking
- Time tracking for performance analysis (`executor.py:137-144`)

**Dependency Management Pattern (`executor.py:162-169`):**
- Check dependencies before node execution
- Skip nodes with unsatisfied dependencies
- Maintain execution order for dependent chains

**Result Tracking Pattern (`executor.py:45-47, 56-71`):**
- Comprehensive result tracking in ExecutionResult class
- Node-level status tracking and summary generation
- Duration and timing information collection

**Error Propagation Pattern (`tests/test_error_handling.py`):**
- Detailed error messages with context
- Chained exception handling with cause tracking
- User-friendly guidance in error messages

## Risks and Mitigations

| Risk | Impact | Mitigation |
| -------- | ----------------- | ---------------- |
| Parallel execution complexity with dependencies | High | Implement topological sorting for safe parallel groups |
| Feedback loop causing infinite replanning cycles | High | Add limits on replanning attempts and cycle detection |
| Breaking backward compatibility with new API | Medium | Extend existing methods with optional parameters |
| Race conditions in parallel node execution | High | Use asyncio synchronization primitives and proper result collection |
| Workflow visualization performance impact | Low | Make visualization optional and cache generated diagrams |
| State persistence complexity | Medium | Start with simple JSON serialization, extend as needed |

## Recommended Approach

**Phase 1: Parallel Execution Foundation**
Extend `WorkflowExecutor.execute_plan()` to use `asyncio.gather()` for executing independent nodes in parallel while preserving dependency order. Implement topological sorting to identify safe parallel execution groups.

**Phase 2: Feedback Mechanism**
Add feedback capability to `ExecutionResult` and create a communication channel from `WorkflowExecutor` back to `MetaPlanner`. Implement basic adaptive replanning when nodes fail with configurable retry limits.

**Phase 3: Progress Callbacks and Visualization**
Add progress callback support to the execution engine and implement ASCII/Mermaid workflow diagram generation. Include real-time execution status visualization.

**Phase 4: Workflow Persistence**
Implement workflow state persistence and resume functionality, enabling long-running workflows to survive interruptions and continue from checkpoints.

This approach builds incrementally on the existing solid foundation while maintaining the clean API and comprehensive error handling patterns established in the codebase.

## Open Questions

- **Parallel Execution Groups**: Should we implement automatic detection of parallelizable node groups or require explicit parallel node definitions?
- **Feedback Granularity**: Should feedback be sent after each node completion, at workflow milestones, or only on failures?
- **Replanning Scope**: When replanning occurs, should we regenerate the entire workflow or only the failed portion?
- **Progress Callback Interface**: What level of detail should progress callbacks provide - node-level, workflow-level, or both?
- **Visualization Format**: Should workflow diagrams be ASCII for simplicity or Mermaid for richer visualization capabilities?
- **Persistence Granularity**: Should we persist at the node level for fine-grained resume capability or workflow level for simplicity?