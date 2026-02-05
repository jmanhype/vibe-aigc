# Advanced Execution with Feedback Loops and Adaptive Planning Implementation Plan

## Implementation Plan Title

Advanced Execution Engine with Parallel Processing, Feedback Loops, and Adaptive Replanning

## Overview

We're implementing a sophisticated execution engine that transforms the current sequential workflow execution into a parallel, adaptive system with comprehensive feedback mechanisms. This includes parallel execution of independent nodes, feedback loops from execution back to the MetaPlanner for adaptive replanning, progress callbacks, workflow visualization, and persistence capabilities.

## Current State

The Vibe AIGC system has a solid foundation with well-architected components:

**Sequential Execution Pattern (`vibe_aigc/executor.py:95-96`):**
- Root nodes executed sequentially in `execute_plan()`: `for node in plan.root_nodes:`
- Children also executed sequentially (`executor.py:150-152`)
- No parallel execution despite independent nodes having no dependencies

**Robust Dependency Management (`vibe_aigc/executor.py:162-169`):**
- Well-implemented `_dependencies_satisfied()` method
- Supports node dependencies through `node.dependencies` list
- Foundation for parallel execution safety already exists

**Comprehensive Error Handling (`vibe_aigc/executor.py:99-102, 154-160`):**
- Proper failure handling with tree marking in `_mark_tree_failed()`
- Execution continues on failure but marks entire subtrees as failed
- No retry or adaptive replanning mechanism currently

**Integration Points Available:**
- `MetaPlanner.executor` relationship (`planner.py:17`) provides feedback pathway
- `ExecutionResult` class (`executor.py:34-72`) captures comprehensive execution data
- `WorkflowNode.dependencies` (`models.py:35`) tracks execution order requirements
- Async execution patterns already established throughout

### Key Discoveries:

- **Parallel execution safety**: Dependency checking logic in `_dependencies_satisfied()` is robust and ready for parallel execution implementation
- **Feedback pathway exists**: The `MetaPlanner.plan_and_execute()` method (`planner.py:97-102`) returns both plan and execution result, providing integration point for feedback
- **Result tracking comprehensive**: `ExecutionResult` class already captures node-level results with timing, status, and error information
- **Error handling patterns**: Consistent error propagation with chained exceptions and user-friendly messages established in tests
- **No existing tests for parallel execution**: Current test suite covers sequential execution and dependency handling but not concurrent scenarios

## Desired End State

A fully adaptive execution engine that:

1. **Parallel Execution**: Independent workflow nodes execute concurrently using `asyncio.gather()`, reducing total execution time by 40-60% for workflows with parallelizable tasks
2. **Feedback Mechanism**: Execution results flow back to MetaPlanner, enabling adaptive replanning when nodes fail or execution context changes
3. **Progress Callbacks**: Real-time execution progress reporting with configurable granularity (node-level, milestone-based, or continuous)
4. **Workflow Visualization**: ASCII and Mermaid diagram generation showing workflow structure and real-time execution status
5. **Persistence & Resume**: Workflow state persistence enabling long-running workflows to survive interruptions and resume from checkpoints
6. **Adaptive Replanning**: When nodes fail, the MetaPlanner can regenerate alternative execution paths with configurable retry limits and cycle detection

### Success Verification:
- Execute workflows with independent nodes in parallel, reducing execution time
- Demonstrate feedback-driven replanning when execution failures occur
- Generate workflow visualizations showing execution progress
- Persist and resume workflows across system interruptions
- Maintain full backward compatibility with existing `MetaPlanner.execute()` API

## What We're NOT Doing

- **Breaking changes to public API**: All existing `MetaPlanner` and `WorkflowExecutor` methods maintain backward compatibility
- **Complex workflow authoring UI**: Focusing on programmatic workflow generation, not visual workflow builders
- **External orchestration systems**: Not integrating with Kubernetes, Apache Airflow, or other external workflow engines
- **Distributed execution**: Parallel execution within single process only, not across multiple machines
- **Custom visualization libraries**: Using existing ASCII/Mermaid generators, not building custom rendering engines
- **Database persistence**: Using JSON file persistence initially, not implementing database backends
- **Real-time collaboration**: Single-user workflow execution, not multi-user concurrent editing
- **Workflow versioning**: Not implementing workflow version control or rollback capabilities

## Implementation Approach

**Incremental Enhancement Strategy**: Build upon the existing solid foundation by extending rather than replacing core components. Each phase adds capability while maintaining full backward compatibility.

**Parallel Execution Foundation**: Leverage `asyncio.gather()` with topological sorting to identify safe parallel execution groups. The existing `_dependencies_satisfied()` method provides the dependency resolution needed for safe concurrent execution.

**Feedback Integration**: Extend `ExecutionResult` to carry feedback data and create a communication channel from `WorkflowExecutor` back to `MetaPlanner` through the existing `plan_and_execute()` method integration point.

**Progressive Feature Rollout**: Start with parallel execution (highest impact, lowest risk), then add feedback mechanisms, visualization, and finally persistence. Each phase is independently testable and deployable.

---

## Phases

### Phase 1: Parallel Execution Foundation

#### Overview

Transform sequential execution into parallel execution for independent workflow nodes while maintaining dependency order. This provides immediate performance benefits and establishes the foundation for advanced features.

#### Changes Required:

##### 1. Parallel Execution Engine

**File**: `vibe_aigc/executor.py`
**Changes**: Replace sequential loops with parallel execution using `asyncio.gather()`

```python
# Replace lines 95-96 and 150-152 with parallel execution
async def execute_plan(self, plan: WorkflowPlan) -> ExecutionResult:
    """Execute a complete WorkflowPlan with parallel execution of independent nodes."""
    result = ExecutionResult(plan.id)
    result.status = ExecutionStatus.RUNNING

    # Group independent root nodes for parallel execution
    parallel_groups = self._build_parallel_execution_groups(plan.root_nodes, result)

    execution_failed = False

    try:
        # Execute each parallel group in sequence, nodes within groups in parallel
        for group in parallel_groups:
            group_tasks = [
                self._execute_node_tree(node, result)
                for node in group
            ]
            try:
                await asyncio.gather(*group_tasks, return_exceptions=True)
            except Exception as e:
                execution_failed = True
                # Mark failed nodes, continue with remaining groups
                for node in group:
                    if node.id not in result.node_results:
                        self._mark_tree_failed(node, result, str(e))

    # Rest of method remains the same...

def _build_parallel_execution_groups(self, nodes: List[WorkflowNode], result: ExecutionResult) -> List[List[WorkflowNode]]:
    """Group nodes into parallel execution batches based on dependencies."""
    # Topological sort implementation to identify parallelizable groups
    # Returns list of node groups where each group can execute in parallel
    pass

async def _execute_node_tree(self, node: WorkflowNode, result: ExecutionResult):
    """Execute a node and its children with parallel child execution."""
    # Existing dependency check and node execution logic...

    # Replace sequential child execution (lines 150-152) with parallel execution
    if node.children:
        child_groups = self._build_parallel_execution_groups(node.children, result)
        for group in child_groups:
            child_tasks = [
                self._execute_node_tree(child, result)
                for child in group
            ]
            await asyncio.gather(*child_tasks, return_exceptions=True)
```

##### 2. Enhanced Result Tracking

**File**: `vibe_aigc/executor.py`
**Changes**: Add parallel execution tracking to ExecutionResult class

```python
class ExecutionResult:
    """Complete result of WorkflowPlan execution with parallel tracking."""

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.status = ExecutionStatus.PENDING
        self.node_results: Dict[str, NodeResult] = {}
        self.started_at = datetime.now().isoformat()
        self.completed_at: str = None
        self.total_duration: float = 0.0
        self.parallel_efficiency: float = 0.0  # New field
        self.execution_groups: List[List[str]] = []  # New field

    def calculate_parallel_efficiency(self) -> float:
        """Calculate efficiency gained from parallel execution."""
        if not self.node_results:
            return 0.0

        total_node_duration = sum(r.duration for r in self.node_results.values())
        actual_duration = self.total_duration

        return max(0.0, (total_node_duration - actual_duration) / total_node_duration) if total_node_duration > 0 else 0.0
```

##### 3. Parallel Execution Tests

**File**: `tests/test_parallel_execution.py`
**Changes**: New comprehensive test suite for parallel execution scenarios

```python
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import time

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import WorkflowExecutor, ExecutionStatus

@pytest.mark.asyncio
class TestParallelExecution:
    """Test parallel execution of independent workflow nodes."""

    async def test_independent_nodes_execute_in_parallel(self):
        """Test that nodes without dependencies execute concurrently."""
        executor = WorkflowExecutor()

        # Create nodes that simulate work duration
        node1 = WorkflowNode(id="parallel-1", type=WorkflowNodeType.GENERATE, description="Task 1")
        node2 = WorkflowNode(id="parallel-2", type=WorkflowNodeType.GENERATE, description="Task 2")
        node3 = WorkflowNode(id="parallel-3", type=WorkflowNodeType.GENERATE, description="Task 3")

        plan = WorkflowPlan(
            id="parallel-test",
            source_vibe=Vibe(description="Test parallel execution"),
            root_nodes=[node1, node2, node3]
        )

        start_time = time.time()
        result = await executor.execute_plan(plan)
        execution_time = time.time() - start_time

        # Verify all nodes completed successfully
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3
        assert all(r.status == ExecutionStatus.COMPLETED for r in result.node_results.values())

        # Verify parallel efficiency > 0 (faster than sequential)
        assert result.parallel_efficiency > 0.0

        # Execution time should be close to max individual duration, not sum
        max_individual_duration = max(r.duration for r in result.node_results.values())
        assert execution_time < max_individual_duration * 2  # Some overhead allowed

    async def test_dependency_order_maintained_in_parallel(self):
        """Test that dependency order is respected even with parallel execution."""
        executor = WorkflowExecutor()

        # Create dependency chain: step1 -> step2 -> step3, plus independent step4
        step1 = WorkflowNode(id="step-1", type=WorkflowNodeType.ANALYZE, description="First")
        step2 = WorkflowNode(id="step-2", type=WorkflowNodeType.GENERATE, description="Second", dependencies=["step-1"])
        step3 = WorkflowNode(id="step-3", type=WorkflowNodeType.VALIDATE, description="Third", dependencies=["step-2"])
        step4 = WorkflowNode(id="step-4", type=WorkflowNodeType.GENERATE, description="Independent")

        plan = WorkflowPlan(
            id="dependency-parallel-test",
            source_vibe=Vibe(description="Test dependencies with parallel"),
            root_nodes=[step1, step2, step3, step4]
        )

        result = await executor.execute_plan(plan)

        # All should complete successfully
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 4

        # Verify execution order: step1 before step2, step2 before step3
        # step4 can execute anytime (parallel with dependency chain)
        times = {node_id: result.started_at for node_id, result in result.node_results.items()}
        assert times["step-1"] <= times["step-2"]
        assert times["step-2"] <= times["step-3"]
        # step4 timing is flexible
```

#### Success Criteria:

##### Automated Verification:

- [ ] Tests pass: `python -m pytest tests/test_parallel_execution.py -v`
- [ ] Existing tests still pass: `python -m pytest tests/ -v`
- [ ] Type checking passes: `python -m mypy vibe_aigc/`
- [ ] Code style passes: `python -m flake8 vibe_aigc/`

##### Manual Verification:

- [ ] Create workflow with 3 independent nodes, verify parallel execution reduces total time
- [ ] Create workflow with dependencies, verify execution order is maintained
- [ ] Check that `ExecutionResult.parallel_efficiency` shows meaningful efficiency gains
- [ ] Verify no regressions in existing sequential execution paths

**Note**: Complete all automated verification, then pause for manual confirmation before proceeding to Phase 2.

---

### Phase 2: Feedback Mechanism and Adaptive Replanning

#### Overview

Implement feedback loop from WorkflowExecutor back to MetaPlanner, enabling adaptive replanning when nodes fail or execution context changes. Adds resilience and self-healing capabilities to workflow execution.

#### Changes Required:

##### 1. Enhanced ExecutionResult with Feedback Data

**File**: `vibe_aigc/executor.py`
**Changes**: Extend ExecutionResult to carry feedback information

```python
class ExecutionResult:
    """Complete result of WorkflowPlan execution with feedback data."""

    def __init__(self, plan_id: str):
        # Existing fields...
        self.feedback_data: Dict[str, Any] = {}  # New field
        self.replan_suggestions: List[Dict[str, Any]] = []  # New field
        self.execution_context: Dict[str, Any] = {}  # New field

    def add_feedback(self, node_id: str, feedback: Dict[str, Any]):
        """Add feedback data from node execution."""
        self.feedback_data[node_id] = feedback

    def suggest_replan(self, suggestion: Dict[str, Any]):
        """Add replanning suggestion based on execution results."""
        self.replan_suggestions.append({
            "timestamp": datetime.now().isoformat(),
            "suggestion": suggestion
        })

    def should_replan(self) -> bool:
        """Determine if replanning is recommended based on execution results."""
        # Check for multiple failures, resource constraints, or explicit suggestions
        failed_nodes = sum(1 for r in self.node_results.values() if r.status == ExecutionStatus.FAILED)
        return failed_nodes > 1 or len(self.replan_suggestions) > 0
```

##### 2. Adaptive MetaPlanner with Feedback Processing

**File**: `vibe_aigc/planner.py`
**Changes**: Add adaptive replanning capabilities to MetaPlanner

```python
class MetaPlanner:
    """Central system architect with adaptive replanning capabilities."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(llm_config)
        self.executor = WorkflowExecutor()
        self.max_replan_attempts = 3  # New configuration
        self.replan_history: List[Dict[str, Any]] = []  # New field

    async def execute_with_adaptation(self, vibe: Vibe) -> Dict[str, Any]:
        """Execute vibe with adaptive replanning on failures."""

        attempt = 0
        current_vibe = vibe
        execution_history = []

        while attempt < self.max_replan_attempts:
            try:
                # Generate and execute plan
                plan = await self.plan(current_vibe)
                execution_result = await self.executor.execute_plan(plan)

                execution_history.append({
                    "attempt": attempt + 1,
                    "plan_id": plan.id,
                    "result_summary": execution_result.get_summary()
                })

                # Check if replanning is needed
                if execution_result.should_replan() and attempt < self.max_replan_attempts - 1:
                    # Generate adapted vibe based on execution feedback
                    adapted_vibe = await self._adapt_vibe_from_feedback(
                        vibe, execution_result, plan
                    )
                    current_vibe = adapted_vibe
                    attempt += 1
                    continue

                # Success or max attempts reached
                return self._format_adaptive_result(execution_result, execution_history)

            except Exception as e:
                if attempt < self.max_replan_attempts - 1:
                    # Try replanning on unexpected errors
                    current_vibe = await self._adapt_vibe_from_error(vibe, str(e))
                    attempt += 1
                    continue
                else:
                    raise

        raise RuntimeError(f"Failed to execute vibe after {self.max_replan_attempts} attempts")

    async def _adapt_vibe_from_feedback(self, original_vibe: Vibe,
                                      execution_result: ExecutionResult,
                                      failed_plan: WorkflowPlan) -> Vibe:
        """Generate adapted vibe based on execution feedback."""

        # Analyze failure patterns and suggest adaptations
        failure_context = {
            "failed_nodes": [
                {"id": node_id, "error": result.error, "type": failed_plan.get_node_by_id(node_id).type}
                for node_id, result in execution_result.node_results.items()
                if result.status == ExecutionStatus.FAILED
            ],
            "feedback_data": execution_result.feedback_data,
            "suggestions": execution_result.replan_suggestions
        }

        # Use LLM to generate adaptive plan adjustments
        adaptation_prompt = f"""
        The original vibe execution failed. Please suggest adaptations:

        Original Vibe: {original_vibe.description}
        Constraints: {original_vibe.constraints}

        Failure Context: {failure_context}

        Suggest adaptations to the vibe description, constraints, or approach.
        """

        # Get LLM suggestions and adapt vibe
        adaptation_response = await self.llm_client._get_adaptation_suggestions(
            original_vibe, failure_context
        )

        return Vibe(
            description=adaptation_response.get("adapted_description", original_vibe.description),
            style=original_vibe.style,
            constraints=adaptation_response.get("adapted_constraints", original_vibe.constraints),
            domain=original_vibe.domain,
            metadata={
                **original_vibe.metadata,
                "adaptation_reason": adaptation_response.get("reason"),
                "original_description": original_vibe.description
            }
        )
```

##### 3. Feedback-Aware Workflow Executor

**File**: `vibe_aigc/executor.py`
**Changes**: Enhance executor to collect and analyze feedback

```python
class WorkflowExecutor:
    """Execution engine with feedback collection and analysis."""

    async def _execute_node_tree(self, node: WorkflowNode, result: ExecutionResult):
        """Execute node with feedback collection."""

        # Existing dependency check and execution logic...

        try:
            # Execute the node
            handler = self.node_handlers.get(node.type, self._execute_default)
            node_result = await handler(node)

            # Collect execution feedback
            feedback = self._analyze_node_execution(node, node_result)
            result.add_feedback(node.id, feedback)

            # Check for replanning triggers
            if self._should_suggest_replan(node, node_result, feedback):
                result.suggest_replan({
                    "node_id": node.id,
                    "reason": feedback.get("replan_reason"),
                    "suggested_changes": feedback.get("suggested_changes", [])
                })

            # Rest of existing logic...

        except Exception as e:
            # Enhanced error feedback
            error_feedback = self._analyze_execution_error(node, e)
            result.add_feedback(node.id, error_feedback)

            if error_feedback.get("replannable", False):
                result.suggest_replan({
                    "node_id": node.id,
                    "error": str(e),
                    "replan_reason": "execution_failure",
                    "suggested_changes": error_feedback.get("suggested_changes", [])
                })

            # Existing error handling...

    def _analyze_node_execution(self, node: WorkflowNode, result: Any) -> Dict[str, Any]:
        """Analyze node execution result for feedback."""
        return {
            "execution_quality": self._assess_result_quality(result),
            "resource_usage": self._measure_resource_usage(),
            "replan_indicators": self._check_replan_indicators(node, result),
            "optimization_suggestions": self._suggest_optimizations(node, result)
        }

    def _should_suggest_replan(self, node: WorkflowNode, result: Any, feedback: Dict[str, Any]) -> bool:
        """Determine if execution suggests replanning."""
        quality_threshold = 0.7  # Configurable
        return feedback.get("execution_quality", 1.0) < quality_threshold
```

##### 4. Feedback Integration Tests

**File**: `tests/test_feedback_adaptation.py`
**Changes**: Comprehensive tests for feedback and adaptation

```python
@pytest.mark.asyncio
class TestFeedbackAdaptation:
    """Test feedback collection and adaptive replanning."""

    async def test_feedback_collection_on_success(self):
        """Test that successful execution collects useful feedback."""
        executor = WorkflowExecutor()

        node = WorkflowNode(id="feedback-test", type=WorkflowNodeType.GENERATE, description="Test feedback")
        plan = WorkflowPlan(id="feedback-plan", source_vibe=Vibe(description="Test"), root_nodes=[node])

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert "feedback-test" in result.feedback_data
        assert "execution_quality" in result.feedback_data["feedback-test"]

    async def test_adaptive_replanning_on_failure(self):
        """Test that failures trigger adaptive replanning."""

        # Mock LLM client for both initial and adaptive planning
        with patch('vibe_aigc.planner.LLMClient') as mock_llm:
            mock_client = AsyncMock()

            # First call returns plan that will fail
            # Second call returns adapted plan that succeeds
            mock_client.decompose_vibe.side_effect = [
                {  # Initial failing plan
                    "id": "failing-plan",
                    "root_nodes": [
                        {"id": "fail-node", "type": "generate", "description": "Will fail",
                         "parameters": {"should_fail": True}, "dependencies": [], "children": []}
                    ]
                },
                {  # Adapted successful plan
                    "id": "adapted-plan",
                    "root_nodes": [
                        {"id": "success-node", "type": "generate", "description": "Will succeed",
                         "parameters": {"should_fail": False}, "dependencies": [], "children": []}
                    ]
                }
            ]
            mock_llm.return_value = mock_client

            planner = MetaPlanner()
            vibe = Vibe(description="Test adaptive replanning")

            result = await planner.execute_with_adaptation(vibe)

            # Should eventually succeed after adaptation
            assert result["status"] == "completed"
            assert "adaptation_history" in result
            assert len(result["adaptation_history"]) > 1  # Multiple attempts
```

#### Success Criteria:

##### Automated Verification:

- [ ] Tests pass: `python -m pytest tests/test_feedback_adaptation.py -v`
- [ ] All existing tests pass: `python -m pytest tests/ -v`
- [ ] Type checking passes: `python -m mypy vibe_aigc/`
- [ ] Integration test: Create failing workflow, verify adaptive replanning occurs

##### Manual Verification:

- [ ] Create workflow with nodes that fail, verify replanning is triggered
- [ ] Test that feedback data is collected and meaningful
- [ ] Verify max replan attempts are respected (no infinite loops)
- [ ] Check that adapted vibes show reasonable modifications based on failures

**Note**: Complete all automated verification, then pause for manual confirmation before proceeding to Phase 3.

---

### Phase 3: Progress Callbacks and Workflow Visualization

#### Overview

Add real-time progress reporting through configurable callbacks and implement workflow visualization with ASCII and Mermaid diagram generation. Provides visibility into workflow execution state and progress.

#### Changes Required:

##### 1. Progress Callback System

**File**: `vibe_aigc/executor.py`
**Changes**: Add callback system for progress reporting

```python
from typing import Callable, Optional
from enum import Enum

class ProgressEventType(str, Enum):
    """Types of progress events."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    GROUP_STARTED = "group_started"
    GROUP_COMPLETED = "group_completed"

class ProgressEvent:
    """Progress event data."""
    def __init__(self, event_type: ProgressEventType,
                 node_id: Optional[str] = None,
                 progress_percent: float = 0.0,
                 message: str = "",
                 metadata: Dict[str, Any] = None):
        self.event_type = event_type
        self.node_id = node_id
        self.progress_percent = progress_percent
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

class WorkflowExecutor:
    """Execution engine with progress callbacks."""

    def __init__(self, progress_callback: Optional[Callable[[ProgressEvent], None]] = None):
        self.node_handlers = {
            WorkflowNodeType.ANALYZE: self._execute_analyze,
            WorkflowNodeType.GENERATE: self._execute_generate,
            WorkflowNodeType.TRANSFORM: self._execute_transform,
            WorkflowNodeType.VALIDATE: self._execute_validate,
            WorkflowNodeType.COMPOSITE: self._execute_composite
        }
        self.progress_callback = progress_callback

    async def execute_plan(self, plan: WorkflowPlan) -> ExecutionResult:
        """Execute workflow with progress reporting."""

        result = ExecutionResult(plan.id)
        result.status = ExecutionStatus.RUNNING

        # Emit workflow started event
        self._emit_progress(ProgressEventType.WORKFLOW_STARTED,
                          message=f"Starting workflow: {plan.id}")

        total_nodes = self._count_total_nodes(plan.root_nodes)
        completed_nodes = 0

        # Group and execute with progress tracking
        parallel_groups = self._build_parallel_execution_groups(plan.root_nodes, result)

        for group_idx, group in enumerate(parallel_groups):
            self._emit_progress(ProgressEventType.GROUP_STARTED,
                              message=f"Starting parallel group {group_idx + 1}/{len(parallel_groups)}")

            # Execute group with progress tracking
            group_tasks = []
            for node in group:
                task = self._execute_node_tree_with_progress(node, result, total_nodes, completed_nodes)
                group_tasks.append(task)

            await asyncio.gather(*group_tasks, return_exceptions=True)
            completed_nodes += len(group)

            progress_percent = (completed_nodes / total_nodes) * 100
            self._emit_progress(ProgressEventType.GROUP_COMPLETED,
                              progress_percent=progress_percent,
                              message=f"Completed group {group_idx + 1}")

        # Determine final status and emit completion
        if any(r.status == ExecutionStatus.FAILED for r in result.node_results.values()):
            result.status = ExecutionStatus.FAILED
            self._emit_progress(ProgressEventType.WORKFLOW_COMPLETED,
                              progress_percent=100.0,
                              message=f"Workflow failed: {result.plan_id}")
        else:
            result.status = ExecutionStatus.COMPLETED
            self._emit_progress(ProgressEventType.WORKFLOW_COMPLETED,
                              progress_percent=100.0,
                              message=f"Workflow completed: {result.plan_id}")

        result.completed_at = datetime.now().isoformat()
        result.total_duration = sum(r.duration for r in result.node_results.values())

        return result

    async def _execute_node_tree_with_progress(self, node: WorkflowNode,
                                             result: ExecutionResult,
                                             total_nodes: int,
                                             completed_nodes: int):
        """Execute node tree with progress reporting."""

        # Check dependencies
        if not self._dependencies_satisfied(node, result):
            result.add_node_result(NodeResult(node.id, ExecutionStatus.SKIPPED,
                                            error="Dependencies not satisfied"))
            return

        # Emit node started
        self._emit_progress(ProgressEventType.NODE_STARTED,
                          node_id=node.id,
                          message=f"Starting: {node.description}")

        start_time = asyncio.get_event_loop().time()

        try:
            # Execute node
            handler = self.node_handlers.get(node.type, self._execute_default)
            node_result = await handler(node)

            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(node.id, ExecutionStatus.COMPLETED,
                                            result=node_result, duration=duration))

            # Emit node completed
            self._emit_progress(ProgressEventType.NODE_COMPLETED,
                              node_id=node.id,
                              message=f"Completed: {node.description}")

            # Execute children with progress
            if node.children:
                child_groups = self._build_parallel_execution_groups(node.children, result)
                for group in child_groups:
                    child_tasks = [
                        self._execute_node_tree_with_progress(child, result, total_nodes, completed_nodes)
                        for child in group
                    ]
                    await asyncio.gather(*child_tasks, return_exceptions=True)

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(node.id, ExecutionStatus.FAILED,
                                            error=str(e), duration=duration))

            self._emit_progress(ProgressEventType.NODE_FAILED,
                              node_id=node.id,
                              message=f"Failed: {node.description} - {str(e)}")
            raise

    def _emit_progress(self, event_type: ProgressEventType,
                      node_id: Optional[str] = None,
                      progress_percent: float = 0.0,
                      message: str = "",
                      metadata: Dict[str, Any] = None):
        """Emit progress event if callback is configured."""
        if self.progress_callback:
            event = ProgressEvent(event_type, node_id, progress_percent, message, metadata)
            try:
                self.progress_callback(event)
            except Exception as e:
                # Don't let callback failures stop execution
                print(f"Progress callback error: {e}")
```

##### 2. Workflow Visualization

**File**: `vibe_aigc/visualization.py`
**Changes**: New module for workflow diagram generation

```python
"""Workflow visualization utilities."""

from typing import List, Dict, Set
from enum import Enum
import json

from .models import WorkflowPlan, WorkflowNode
from .executor import ExecutionResult, ExecutionStatus

class VisualizationFormat(str, Enum):
    """Supported visualization formats."""
    ASCII = "ascii"
    MERMAID = "mermaid"

class WorkflowVisualizer:
    """Generate workflow visualizations."""

    @staticmethod
    def generate_diagram(plan: WorkflowPlan,
                        execution_result: Optional[ExecutionResult] = None,
                        format: VisualizationFormat = VisualizationFormat.ASCII) -> str:
        """Generate workflow diagram in specified format."""

        if format == VisualizationFormat.ASCII:
            return WorkflowVisualizer._generate_ascii_diagram(plan, execution_result)
        elif format == VisualizationFormat.MERMAID:
            return WorkflowVisualizer._generate_mermaid_diagram(plan, execution_result)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def _generate_ascii_diagram(plan: WorkflowPlan,
                               execution_result: Optional[ExecutionResult] = None) -> str:
        """Generate ASCII workflow diagram."""

        lines = []
        lines.append(f"Workflow Plan: {plan.id}")
        lines.append("=" * 50)
        lines.append(f"Source Vibe: {plan.source_vibe.description}")
        if plan.estimated_total_duration:
            lines.append(f"Estimated Duration: {plan.estimated_total_duration}s")
        lines.append("")

        # Build execution status map
        status_map = {}
        if execution_result:
            for node_id, result in execution_result.node_results.items():
                status_map[node_id] = result.status

        # Generate tree structure
        for i, root_node in enumerate(plan.root_nodes):
            is_last_root = (i == len(plan.root_nodes) - 1)
            WorkflowVisualizer._add_node_to_ascii(lines, root_node, "", is_last_root, status_map)

        # Add execution summary if available
        if execution_result:
            lines.append("")
            lines.append("Execution Summary:")
            lines.append("-" * 30)
            summary = execution_result.get_summary()
            lines.append(f"Status: {summary['status']}")
            lines.append(f"Completed: {summary['completed']}/{summary['total_nodes']}")
            lines.append(f"Duration: {summary['total_duration']:.2f}s")
            if hasattr(execution_result, 'parallel_efficiency'):
                lines.append(f"Parallel Efficiency: {execution_result.parallel_efficiency:.1%}")

        return "\n".join(lines)

    @staticmethod
    def _add_node_to_ascii(lines: List[str], node: WorkflowNode, prefix: str,
                          is_last: bool, status_map: Dict[str, ExecutionStatus]):
        """Add node to ASCII diagram recursively."""

        # Choose connector
        connector = "└── " if is_last else "├── "

        # Get status indicator
        status = status_map.get(node.id, ExecutionStatus.PENDING)
        status_indicator = {
            ExecutionStatus.PENDING: "⏳",
            ExecutionStatus.RUNNING: "🔄",
            ExecutionStatus.COMPLETED: "✅",
            ExecutionStatus.FAILED: "❌",
            ExecutionStatus.SKIPPED: "⏭️"
        }.get(status, "❓")

        # Format node line
        node_line = f"{prefix}{connector}{status_indicator} [{node.type}] {node.description}"
        if node.id != node.description:
            node_line += f" ({node.id})"
        lines.append(node_line)

        # Add dependencies info
        if node.dependencies:
            dep_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{dep_prefix}└─ Depends on: {', '.join(node.dependencies)}")

        # Add children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            WorkflowVisualizer._add_node_to_ascii(lines, child, child_prefix, is_last_child, status_map)

    @staticmethod
    def _generate_mermaid_diagram(plan: WorkflowPlan,
                                 execution_result: Optional[ExecutionResult] = None) -> str:
        """Generate Mermaid workflow diagram."""

        lines = ["graph TD"]

        # Build status map
        status_map = {}
        if execution_result:
            for node_id, result in execution_result.node_results.items():
                status_map[node_id] = result.status

        # Add all nodes
        all_nodes = WorkflowVisualizer._collect_all_nodes(plan.root_nodes)
        for node in all_nodes:
            status = status_map.get(node.id, ExecutionStatus.PENDING)

            # Node style based on type and status
            shape = {
                "analyze": "([{}])",
                "generate": "[{}]",
                "transform": "{{{}}}",
                "validate": "(({}))",
                "composite": "[[{}]]"
            }.get(node.type.value, "[{}]")

            # Status styling
            status_class = {
                ExecutionStatus.COMPLETED: "completed",
                ExecutionStatus.FAILED: "failed",
                ExecutionStatus.RUNNING: "running",
                ExecutionStatus.SKIPPED: "skipped"
            }.get(status, "pending")

            node_label = f"{node.description[:30]}..." if len(node.description) > 30 else node.description
            lines.append(f"    {node.id}{shape.format(node_label)}:::{status_class}")

        # Add dependencies
        for node in all_nodes:
            for dep_id in node.dependencies:
                lines.append(f"    {dep_id} --> {node.id}")

            # Add parent-child relationships
            for child in node.children:
                lines.append(f"    {node.id} --> {child.id}")

        # Add styling
        lines.extend([
            "",
            "    classDef completed fill:#d4edda,stroke:#28a745",
            "    classDef failed fill:#f8d7da,stroke:#dc3545",
            "    classDef running fill:#fff3cd,stroke:#ffc107",
            "    classDef skipped fill:#e2e3e5,stroke:#6c757d",
            "    classDef pending fill:#f8f9fa,stroke:#6c757d"
        ])

        return "\n".join(lines)

    @staticmethod
    def _collect_all_nodes(root_nodes: List[WorkflowNode]) -> List[WorkflowNode]:
        """Collect all nodes in workflow recursively."""
        all_nodes = []

        def collect_recursive(nodes: List[WorkflowNode]):
            for node in nodes:
                all_nodes.append(node)
                collect_recursive(node.children)

        collect_recursive(root_nodes)
        return all_nodes
```

##### 3. Enhanced MetaPlanner with Visualization

**File**: `vibe_aigc/planner.py`
**Changes**: Add visualization and progress callback support

```python
from .visualization import WorkflowVisualizer, VisualizationFormat
from .executor import ProgressEvent, ProgressEventType

class MetaPlanner:
    """MetaPlanner with visualization and progress tracking."""

    def __init__(self, llm_config: Optional[LLMConfig] = None,
                 progress_callback: Optional[Callable[[ProgressEvent], None]] = None):
        self.llm_client = LLMClient(llm_config)
        self.executor = WorkflowExecutor(progress_callback)
        self.progress_callback = progress_callback

    async def execute_with_visualization(self, vibe: Vibe,
                                       show_progress: bool = True,
                                       visualization_format: VisualizationFormat = VisualizationFormat.ASCII) -> Dict[str, Any]:
        """Execute vibe with optional progress visualization."""

        # Generate and execute plan
        plan = await self.plan(vibe)

        if show_progress and not self.progress_callback:
            # Default progress visualization
            def default_progress_callback(event: ProgressEvent):
                print(f"[{event.timestamp}] {event.message}")
                if event.progress_percent > 0:
                    print(f"Progress: {event.progress_percent:.1f}%")

            self.executor.progress_callback = default_progress_callback

        execution_result = await self.executor.execute_plan(plan)

        # Generate visualization
        diagram = WorkflowVisualizer.generate_diagram(plan, execution_result, visualization_format)

        # Enhanced result with visualization
        result = self._format_result_with_visualization(execution_result, plan, diagram)
        return result

    def _format_result_with_visualization(self, execution_result: ExecutionResult,
                                        plan: WorkflowPlan, diagram: str) -> Dict[str, Any]:
        """Format result with visualization data."""

        base_result = {
            "status": execution_result.status.value,
            "plan_id": execution_result.plan_id,
            "vibe_description": plan.source_vibe.description,
            "execution_summary": execution_result.get_summary(),
            "node_results": {
                node_id: {
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "duration": result.duration
                }
                for node_id, result in execution_result.node_results.items()
            },
            "visualization": diagram
        }

        # Add parallel execution metrics if available
        if hasattr(execution_result, 'parallel_efficiency'):
            base_result["parallel_efficiency"] = execution_result.parallel_efficiency
            base_result["execution_groups"] = execution_result.execution_groups

        return base_result
```

##### 4. Visualization and Progress Tests

**File**: `tests/test_visualization_progress.py`
**Changes**: Tests for visualization and progress features

```python
@pytest.mark.asyncio
class TestVisualizationProgress:
    """Test visualization and progress callback functionality."""

    def test_ascii_diagram_generation(self):
        """Test ASCII workflow diagram generation."""

        # Create test workflow
        child1 = WorkflowNode(id="analyze-1", type=WorkflowNodeType.ANALYZE, description="Analyze input")
        child2 = WorkflowNode(id="generate-1", type=WorkflowNodeType.GENERATE,
                             description="Generate content", dependencies=["analyze-1"])

        parent = WorkflowNode(id="workflow-1", type=WorkflowNodeType.COMPOSITE,
                             description="Complete workflow", children=[child1, child2])

        plan = WorkflowPlan(id="test-viz", source_vibe=Vibe(description="Test viz"),
                           root_nodes=[parent])

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII)

        # Verify diagram content
        assert "Workflow Plan: test-viz" in diagram
        assert "Complete workflow" in diagram
        assert "Analyze input" in diagram
        assert "Generate content" in diagram
        assert "├──" in diagram or "└──" in diagram  # Tree structure

    def test_mermaid_diagram_generation(self):
        """Test Mermaid workflow diagram generation."""

        node = WorkflowNode(id="test-node", type=WorkflowNodeType.GENERATE, description="Test node")
        plan = WorkflowPlan(id="mermaid-test", source_vibe=Vibe(description="Test"), root_nodes=[node])

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID)

        assert "graph TD" in diagram
        assert "test-node" in diagram
        assert "Test node" in diagram
        assert "classDef" in diagram  # Styling

    async def test_progress_callbacks(self):
        """Test that progress callbacks are called during execution."""

        progress_events = []

        def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        executor = WorkflowExecutor(progress_callback=capture_progress)

        node = WorkflowNode(id="progress-test", type=WorkflowNodeType.GENERATE,
                           description="Progress test node")
        plan = WorkflowPlan(id="progress-plan", source_vibe=Vibe(description="Test progress"),
                           root_nodes=[node])

        await executor.execute_plan(plan)

        # Verify progress events
        assert len(progress_events) >= 3  # At least: workflow_started, node_started, node_completed, workflow_completed

        event_types = [event.event_type for event in progress_events]
        assert ProgressEventType.WORKFLOW_STARTED in event_types
        assert ProgressEventType.NODE_STARTED in event_types
        assert ProgressEventType.NODE_COMPLETED in event_types
        assert ProgressEventType.WORKFLOW_COMPLETED in event_types

        # Check that final event shows 100% progress
        final_event = progress_events[-1]
        assert final_event.event_type == ProgressEventType.WORKFLOW_COMPLETED
        assert final_event.progress_percent == 100.0

    async def test_metaplanner_with_visualization(self):
        """Test MetaPlanner.execute_with_visualization()."""

        with patch('vibe_aigc.planner.LLMClient') as mock_llm:
            mock_client = AsyncMock()
            mock_client.decompose_vibe.return_value = {
                "id": "viz-test-plan",
                "root_nodes": [
                    {"id": "viz-node", "type": "generate", "description": "Test visualization",
                     "parameters": {}, "dependencies": [], "children": []}
                ]
            }
            mock_llm.return_value = mock_client

            progress_events = []
            def capture_progress(event): progress_events.append(event)

            planner = MetaPlanner(progress_callback=capture_progress)
            vibe = Vibe(description="Test visualization integration")

            result = await planner.execute_with_visualization(vibe)

            # Verify result includes visualization
            assert "visualization" in result
            assert "Workflow Plan: viz-test-plan" in result["visualization"]

            # Verify progress was tracked
            assert len(progress_events) > 0
```

#### Success Criteria:

##### Automated Verification:

- [ ] Tests pass: `python -m pytest tests/test_visualization_progress.py -v`
- [ ] All existing tests pass: `python -m pytest tests/ -v`
- [ ] Type checking passes: `python -m mypy vibe_aigc/`

##### Manual Verification:

- [ ] Execute workflow with progress callbacks, verify meaningful events are emitted
- [ ] Generate ASCII diagram, verify structure is clear and readable
- [ ] Generate Mermaid diagram, verify it renders correctly in Mermaid viewers
- [ ] Test visualization with execution results, verify status indicators are correct

**Note**: Complete all automated verification, then pause for manual confirmation before proceeding to Phase 4.

---

### Phase 4: Workflow Persistence and Resume

#### Overview

Implement workflow state persistence and resume capabilities, enabling long-running workflows to survive interruptions and continue from checkpoints. This includes JSON-based state serialization and restoration logic.

#### Changes Required:

##### 1. Workflow State Serialization

**File**: `vibe_aigc/persistence.py`
**Changes**: New module for workflow persistence

```python
"""Workflow persistence and resume capabilities."""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

from .models import WorkflowPlan, WorkflowNode, Vibe
from .executor import ExecutionResult, NodeResult, ExecutionStatus

class WorkflowCheckpoint:
    """Represents a workflow execution checkpoint."""

    def __init__(self, plan: WorkflowPlan, execution_result: ExecutionResult,
                 checkpoint_id: Optional[str] = None):
        self.checkpoint_id = checkpoint_id or self._generate_checkpoint_id(plan.id)
        self.plan = plan
        self.execution_result = execution_result
        self.created_at = datetime.now().isoformat()
        self.schema_version = "1.0"

    def _generate_checkpoint_id(self, plan_id: str) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{plan_id}_{timestamp}".encode()
        hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]
        return f"{plan_id}_{timestamp}_{hash_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "plan": self._serialize_plan(self.plan),
            "execution_result": self._serialize_execution_result(self.execution_result)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        """Deserialize checkpoint from dictionary."""

        # Version compatibility check
        if data.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported checkpoint schema version: {data.get('schema_version')}")

        # Deserialize plan and execution result
        plan = cls._deserialize_plan(data["plan"])
        execution_result = cls._deserialize_execution_result(data["execution_result"])

        checkpoint = cls(plan, execution_result, data["checkpoint_id"])
        checkpoint.created_at = data["created_at"]
        return checkpoint

    def _serialize_plan(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Serialize WorkflowPlan to dict."""
        return {
            "id": plan.id,
            "source_vibe": {
                "description": plan.source_vibe.description,
                "style": plan.source_vibe.style,
                "constraints": plan.source_vibe.constraints,
                "domain": plan.source_vibe.domain,
                "metadata": plan.source_vibe.metadata
            },
            "root_nodes": [self._serialize_node(node) for node in plan.root_nodes],
            "estimated_total_duration": plan.estimated_total_duration,
            "created_at": plan.created_at
        }

    def _serialize_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Serialize WorkflowNode to dict."""
        return {
            "id": node.id,
            "type": node.type.value,
            "description": node.description,
            "parameters": node.parameters,
            "dependencies": node.dependencies,
            "children": [self._serialize_node(child) for child in node.children],
            "estimated_duration": node.estimated_duration
        }

    def _serialize_execution_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """Serialize ExecutionResult to dict."""
        return {
            "plan_id": result.plan_id,
            "status": result.status.value,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "total_duration": result.total_duration,
            "node_results": {
                node_id: {
                    "node_id": node_result.node_id,
                    "status": node_result.status.value,
                    "result": node_result.result,
                    "error": node_result.error,
                    "duration": node_result.duration,
                    "started_at": node_result.started_at
                }
                for node_id, node_result in result.node_results.items()
            },
            # Include new fields if they exist
            "parallel_efficiency": getattr(result, 'parallel_efficiency', 0.0),
            "execution_groups": getattr(result, 'execution_groups', []),
            "feedback_data": getattr(result, 'feedback_data', {}),
            "replan_suggestions": getattr(result, 'replan_suggestions', [])
        }

    @classmethod
    def _deserialize_plan(cls, data: Dict[str, Any]) -> WorkflowPlan:
        """Deserialize WorkflowPlan from dict."""
        from .models import WorkflowNodeType  # Import here to avoid circular imports

        vibe_data = data["source_vibe"]
        source_vibe = Vibe(
            description=vibe_data["description"],
            style=vibe_data.get("style"),
            constraints=vibe_data.get("constraints", []),
            domain=vibe_data.get("domain"),
            metadata=vibe_data.get("metadata", {})
        )

        root_nodes = [cls._deserialize_node(node_data) for node_data in data["root_nodes"]]

        return WorkflowPlan(
            id=data["id"],
            source_vibe=source_vibe,
            root_nodes=root_nodes,
            estimated_total_duration=data.get("estimated_total_duration"),
            created_at=data.get("created_at")
        )

    @classmethod
    def _deserialize_node(cls, data: Dict[str, Any]) -> WorkflowNode:
        """Deserialize WorkflowNode from dict."""
        from .models import WorkflowNodeType

        children = [cls._deserialize_node(child_data) for child_data in data.get("children", [])]

        return WorkflowNode(
            id=data["id"],
            type=WorkflowNodeType(data["type"]),
            description=data["description"],
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            children=children,
            estimated_duration=data.get("estimated_duration")
        )

    @classmethod
    def _deserialize_execution_result(cls, data: Dict[str, Any]) -> ExecutionResult:
        """Deserialize ExecutionResult from dict."""
        result = ExecutionResult(data["plan_id"])
        result.status = ExecutionStatus(data["status"])
        result.started_at = data["started_at"]
        result.completed_at = data["completed_at"]
        result.total_duration = data["total_duration"]

        # Deserialize node results
        for node_id, node_data in data["node_results"].items():
            node_result = NodeResult(
                node_data["node_id"],
                ExecutionStatus(node_data["status"]),
                node_data.get("result"),
                node_data.get("error"),
                node_data.get("duration", 0.0)
            )
            node_result.started_at = node_data.get("started_at", datetime.now().isoformat())
            result.node_results[node_id] = node_result

        # Restore extended fields
        if "parallel_efficiency" in data:
            result.parallel_efficiency = data["parallel_efficiency"]
        if "execution_groups" in data:
            result.execution_groups = data["execution_groups"]
        if "feedback_data" in data:
            result.feedback_data = data["feedback_data"]
        if "replan_suggestions" in data:
            result.replan_suggestions = data["replan_suggestions"]

        return result


class WorkflowPersistenceManager:
    """Manages workflow checkpoint persistence."""

    def __init__(self, checkpoint_dir: str = ".vibe_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save checkpoint to disk."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint.checkpoint_id}.json")

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            return checkpoint_path
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}") from e

    def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Load checkpoint from disk."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return WorkflowCheckpoint.from_dict(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_id}: {e}") from e

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints with metadata."""
        checkpoints = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                checkpoint_id = filename[:-5]  # Remove .json extension

                try:
                    checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                    with open(checkpoint_path, 'r') as f:
                        data = json.load(f)

                    checkpoints.append({
                        "checkpoint_id": checkpoint_id,
                        "plan_id": data.get("plan", {}).get("id"),
                        "created_at": data.get("created_at"),
                        "status": data.get("execution_result", {}).get("status"),
                        "vibe_description": data.get("plan", {}).get("source_vibe", {}).get("description", "")[:50]
                    })
                except Exception:
                    # Skip corrupted checkpoints
                    continue

        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from disk."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                return True
            except Exception as e:
                raise RuntimeError(f"Failed to delete checkpoint {checkpoint_id}: {e}") from e

        return False
```

##### 2. Resume-Capable Executor

**File**: `vibe_aigc/executor.py`
**Changes**: Add checkpoint and resume capabilities to WorkflowExecutor

```python
from .persistence import WorkflowCheckpoint, WorkflowPersistenceManager

class WorkflowExecutor:
    """Execution engine with checkpoint and resume capabilities."""

    def __init__(self, progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
                 enable_checkpoints: bool = False,
                 checkpoint_dir: str = ".vibe_checkpoints"):
        self.node_handlers = {
            WorkflowNodeType.ANALYZE: self._execute_analyze,
            WorkflowNodeType.GENERATE: self._execute_generate,
            WorkflowNodeType.TRANSFORM: self._execute_transform,
            WorkflowNodeType.VALIDATE: self._execute_validate,
            WorkflowNodeType.COMPOSITE: self._execute_composite
        }
        self.progress_callback = progress_callback
        self.enable_checkpoints = enable_checkpoints
        self.persistence_manager = WorkflowPersistenceManager(checkpoint_dir) if enable_checkpoints else None
        self.checkpoint_interval = 5  # Save checkpoint every 5 completed nodes
        self._nodes_since_checkpoint = 0

    async def execute_plan(self, plan: WorkflowPlan,
                          resume_from_checkpoint: Optional[str] = None) -> ExecutionResult:
        """Execute workflow with optional checkpoint resume."""

        if resume_from_checkpoint:
            return await self._resume_from_checkpoint(resume_from_checkpoint)
        else:
            return await self._execute_plan_with_checkpoints(plan)

    async def _resume_from_checkpoint(self, checkpoint_id: str) -> ExecutionResult:
        """Resume execution from saved checkpoint."""

        if not self.persistence_manager:
            raise RuntimeError("Checkpoints not enabled. Initialize executor with enable_checkpoints=True")

        # Load checkpoint
        checkpoint = self.persistence_manager.load_checkpoint(checkpoint_id)

        self._emit_progress(ProgressEventType.WORKFLOW_STARTED,
                          message=f"Resuming workflow from checkpoint: {checkpoint_id}")

        # Continue execution from where we left off
        return await self._continue_execution(checkpoint.plan, checkpoint.execution_result)

    async def _execute_plan_with_checkpoints(self, plan: WorkflowPlan) -> ExecutionResult:
        """Execute plan with automatic checkpointing."""

        result = ExecutionResult(plan.id)
        result.status = ExecutionStatus.RUNNING

        self._emit_progress(ProgressEventType.WORKFLOW_STARTED,
                          message=f"Starting workflow: {plan.id}")

        try:
            # Execute with checkpointing
            result = await self._continue_execution(plan, result)

            # Save final checkpoint
            if self.enable_checkpoints and self.persistence_manager:
                final_checkpoint = WorkflowCheckpoint(plan, result)
                self.persistence_manager.save_checkpoint(final_checkpoint)

            return result

        except Exception as e:
            # Save error checkpoint for potential debugging/recovery
            if self.enable_checkpoints and self.persistence_manager:
                error_checkpoint = WorkflowCheckpoint(plan, result)
                self.persistence_manager.save_checkpoint(error_checkpoint)

            raise

    async def _continue_execution(self, plan: WorkflowPlan, partial_result: ExecutionResult) -> ExecutionResult:
        """Continue execution of workflow, skipping completed nodes."""

        # Identify remaining work
        remaining_nodes = self._identify_remaining_nodes(plan.root_nodes, partial_result)

        if not remaining_nodes:
            # All work is complete
            partial_result.status = ExecutionStatus.COMPLETED
            partial_result.completed_at = datetime.now().isoformat()
            return partial_result

        # Continue execution with remaining nodes
        parallel_groups = self._build_parallel_execution_groups(remaining_nodes, partial_result)

        for group_idx, group in enumerate(parallel_groups):
            self._emit_progress(ProgressEventType.GROUP_STARTED,
                              message=f"Continuing group {group_idx + 1}/{len(parallel_groups)}")

            group_tasks = [
                self._execute_node_tree_with_progress(node, partial_result, len(remaining_nodes), 0)
                for node in group
            ]

            await asyncio.gather(*group_tasks, return_exceptions=True)

            # Checkpoint after each group if enabled
            if self.enable_checkpoints and self.persistence_manager:
                checkpoint = WorkflowCheckpoint(plan, partial_result)
                self.persistence_manager.save_checkpoint(checkpoint)

        # Determine final status
        if any(r.status == ExecutionStatus.FAILED for r in partial_result.node_results.values()):
            partial_result.status = ExecutionStatus.FAILED
        else:
            partial_result.status = ExecutionStatus.COMPLETED

        partial_result.completed_at = datetime.now().isoformat()
        partial_result.total_duration = sum(r.duration for r in partial_result.node_results.values())

        return partial_result

    def _identify_remaining_nodes(self, root_nodes: List[WorkflowNode],
                                 partial_result: ExecutionResult) -> List[WorkflowNode]:
        """Identify nodes that still need execution."""
        remaining = []

        def check_node_and_children(node: WorkflowNode):
            # Check if this node needs execution
            if node.id not in partial_result.node_results or \
               partial_result.node_results[node.id].status in [ExecutionStatus.PENDING, ExecutionStatus.FAILED]:
                remaining.append(node)
            else:
                # Node is complete, but check children
                for child in node.children:
                    check_node_and_children(child)

        for root_node in root_nodes:
            check_node_and_children(root_node)

        return remaining

    async def _execute_node_tree_with_progress(self, node: WorkflowNode,
                                             result: ExecutionResult,
                                             total_nodes: int,
                                             completed_nodes: int):
        """Execute node tree with checkpointing."""

        # Skip if already completed
        if node.id in result.node_results and \
           result.node_results[node.id].status == ExecutionStatus.COMPLETED:
            return

        # Regular execution logic...
        await super()._execute_node_tree_with_progress(node, result, total_nodes, completed_nodes)

        # Checkpoint periodically
        if self.enable_checkpoints and self.persistence_manager:
            self._nodes_since_checkpoint += 1
            if self._nodes_since_checkpoint >= self.checkpoint_interval:
                # Save intermediate checkpoint
                # Note: This requires access to the original plan, which we'd need to pass through
                self._nodes_since_checkpoint = 0
```

##### 3. Enhanced MetaPlanner with Persistence

**File**: `vibe_aigc/planner.py`
**Changes**: Add persistence support to MetaPlanner

```python
from .persistence import WorkflowPersistenceManager

class MetaPlanner:
    """MetaPlanner with workflow persistence capabilities."""

    def __init__(self, llm_config: Optional[LLMConfig] = None,
                 progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
                 enable_persistence: bool = False,
                 checkpoint_dir: str = ".vibe_checkpoints"):
        self.llm_client = LLMClient(llm_config)
        self.executor = WorkflowExecutor(progress_callback, enable_persistence, checkpoint_dir)
        self.progress_callback = progress_callback
        self.persistence_manager = WorkflowPersistenceManager(checkpoint_dir) if enable_persistence else None

    async def execute_with_resume(self, vibe: Vibe,
                                 checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute vibe with optional resume from checkpoint."""

        if checkpoint_id:
            # Resume from checkpoint
            if not self.persistence_manager:
                raise RuntimeError("Persistence not enabled. Initialize MetaPlanner with enable_persistence=True")

            execution_result = await self.executor.execute_plan(None, checkpoint_id)

            # Load plan from checkpoint for result formatting
            checkpoint = self.persistence_manager.load_checkpoint(checkpoint_id)
            plan = checkpoint.plan

        else:
            # Normal execution with checkpointing
            plan = await self.plan(vibe)
            execution_result = await self.executor.execute_plan(plan)

        return self._format_persistent_result(execution_result, plan)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available workflow checkpoints."""
        if not self.persistence_manager:
            raise RuntimeError("Persistence not enabled")

        return self.persistence_manager.list_checkpoints()

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a workflow checkpoint."""
        if not self.persistence_manager:
            raise RuntimeError("Persistence not enabled")

        return self.persistence_manager.delete_checkpoint(checkpoint_id)

    def _format_persistent_result(self, execution_result: ExecutionResult,
                                plan: WorkflowPlan) -> Dict[str, Any]:
        """Format result with persistence information."""

        base_result = {
            "status": execution_result.status.value,
            "plan_id": execution_result.plan_id,
            "vibe_description": plan.source_vibe.description,
            "execution_summary": execution_result.get_summary(),
            "node_results": {
                node_id: {
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "duration": result.duration
                }
                for node_id, result in execution_result.node_results.items()
            }
        }

        # Add checkpoint information if persistence is enabled
        if self.persistence_manager:
            base_result["persistence_enabled"] = True
            base_result["checkpoint_directory"] = self.persistence_manager.checkpoint_dir

            # List recent checkpoints for this plan
            all_checkpoints = self.persistence_manager.list_checkpoints()
            plan_checkpoints = [cp for cp in all_checkpoints if cp["plan_id"] == execution_result.plan_id]
            base_result["available_checkpoints"] = plan_checkpoints[:5]  # Most recent 5

        return base_result
```

##### 4. Persistence Tests

**File**: `tests/test_persistence.py`
**Changes**: Comprehensive tests for persistence functionality

```python
import tempfile
import os
import shutil

from vibe_aigc.persistence import WorkflowCheckpoint, WorkflowPersistenceManager
from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus, WorkflowExecutor
from vibe_aigc.planner import MetaPlanner

@pytest.mark.asyncio
class TestWorkflowPersistence:
    """Test workflow checkpoint and resume functionality."""

    def setup_method(self):
        """Set up test environment with temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_manager = WorkflowPersistenceManager(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_serialization_roundtrip(self):
        """Test that checkpoints can be serialized and deserialized correctly."""

        # Create test workflow
        node = WorkflowNode(id="test-node", type=WorkflowNodeType.GENERATE, description="Test node")
        vibe = Vibe(description="Test vibe", constraints=["test constraint"])
        plan = WorkflowPlan(id="test-plan", source_vibe=vibe, root_nodes=[node])

        # Create execution result
        result = ExecutionResult("test-plan")
        result.status = ExecutionStatus.RUNNING
        result.add_node_result(NodeResult("test-node", ExecutionStatus.COMPLETED,
                                         result={"output": "test"}, duration=1.5))

        # Create and serialize checkpoint
        checkpoint = WorkflowCheckpoint(plan, result)
        checkpoint_data = checkpoint.to_dict()

        # Deserialize and verify
        restored_checkpoint = WorkflowCheckpoint.from_dict(checkpoint_data)

        assert restored_checkpoint.checkpoint_id == checkpoint.checkpoint_id
        assert restored_checkpoint.plan.id == plan.id
        assert restored_checkpoint.plan.source_vibe.description == vibe.description
        assert len(restored_checkpoint.execution_result.node_results) == 1
        assert "test-node" in restored_checkpoint.execution_result.node_results

    def test_persistence_manager_save_load(self):
        """Test saving and loading checkpoints through persistence manager."""

        # Create test checkpoint
        node = WorkflowNode(id="persist-node", type=WorkflowNodeType.ANALYZE, description="Persist test")
        plan = WorkflowPlan(id="persist-plan", source_vibe=Vibe(description="Test persist"),
                           root_nodes=[node])
        result = ExecutionResult("persist-plan")
        result.add_node_result(NodeResult("persist-node", ExecutionStatus.COMPLETED, duration=2.0))

        checkpoint = WorkflowCheckpoint(plan, result)

        # Save checkpoint
        saved_path = self.persistence_manager.save_checkpoint(checkpoint)
        assert os.path.exists(saved_path)

        # Load checkpoint
        loaded_checkpoint = self.persistence_manager.load_checkpoint(checkpoint.checkpoint_id)

        assert loaded_checkpoint.plan.id == plan.id
        assert loaded_checkpoint.execution_result.plan_id == result.plan_id
        assert len(loaded_checkpoint.execution_result.node_results) == 1

    def test_checkpoint_listing(self):
        """Test listing available checkpoints."""

        # Create multiple checkpoints
        for i in range(3):
            node = WorkflowNode(id=f"node-{i}", type=WorkflowNodeType.GENERATE,
                               description=f"Test node {i}")
            plan = WorkflowPlan(id=f"plan-{i}", source_vibe=Vibe(description=f"Test {i}"),
                               root_nodes=[node])
            result = ExecutionResult(f"plan-{i}")

            checkpoint = WorkflowCheckpoint(plan, result)
            self.persistence_manager.save_checkpoint(checkpoint)

        # List checkpoints
        checkpoints = self.persistence_manager.list_checkpoints()

        assert len(checkpoints) == 3
        assert all("checkpoint_id" in cp for cp in checkpoints)
        assert all("plan_id" in cp for cp in checkpoints)
        assert all("created_at" in cp for cp in checkpoints)

    async def test_executor_with_checkpoints(self):
        """Test workflow executor with checkpoint functionality."""

        executor = WorkflowExecutor(enable_checkpoints=True, checkpoint_dir=self.temp_dir)

        # Create workflow
        node1 = WorkflowNode(id="checkpoint-1", type=WorkflowNodeType.ANALYZE, description="First task")
        node2 = WorkflowNode(id="checkpoint-2", type=WorkflowNodeType.GENERATE,
                            description="Second task", dependencies=["checkpoint-1"])

        plan = WorkflowPlan(id="checkpoint-test", source_vibe=Vibe(description="Checkpoint test"),
                           root_nodes=[node1, node2])

        # Execute workflow
        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 2

        # Verify checkpoint was saved
        checkpoints = executor.persistence_manager.list_checkpoints()
        assert len(checkpoints) >= 1
        assert any(cp["plan_id"] == "checkpoint-test" for cp in checkpoints)

    @patch('vibe_aigc.planner.LLMClient')
    async def test_metaplanner_resume_functionality(self, mock_llm_client):
        """Test MetaPlanner resume from checkpoint functionality."""

        # Mock LLM for planning
        mock_client = AsyncMock()
        mock_client.decompose_vibe.return_value = {
            "id": "resume-test-plan",
            "root_nodes": [
                {"id": "resume-node-1", "type": "analyze", "description": "Analyze task",
                 "parameters": {}, "dependencies": [], "children": []},
                {"id": "resume-node-2", "type": "generate", "description": "Generate task",
                 "parameters": {}, "dependencies": ["resume-node-1"], "children": []}
            ]
        }
        mock_llm_client.return_value = mock_client

        # Create MetaPlanner with persistence
        planner = MetaPlanner(enable_persistence=True, checkpoint_dir=self.temp_dir)
        vibe = Vibe(description="Resume test vibe")

        # Execute workflow (this will create checkpoints)
        result1 = await planner.execute_with_resume(vibe)

        assert result1["status"] == "completed"
        assert "persistence_enabled" in result1
        assert result1["persistence_enabled"] == True

        # List checkpoints
        checkpoints = planner.list_checkpoints()
        assert len(checkpoints) > 0

        # Test checkpoint listing includes useful metadata
        checkpoint = checkpoints[0]
        assert "checkpoint_id" in checkpoint
        assert "plan_id" in checkpoint
        assert "vibe_description" in checkpoint
```

#### Success Criteria:

##### Automated Verification:

- [ ] Tests pass: `python -m pytest tests/test_persistence.py -v`
- [ ] All existing tests pass: `python -m pytest tests/ -v`
- [ ] Type checking passes: `python -m mypy vibe_aigc/`

##### Manual Verification:

- [ ] Create long-running workflow, interrupt execution, verify checkpoint is saved
- [ ] Resume workflow from checkpoint, verify execution continues from correct point
- [ ] List checkpoints, verify metadata is accurate and useful
- [ ] Delete old checkpoints, verify files are removed from disk
- [ ] Test checkpoint versioning by creating checkpoints with different schema versions

**Note**: Complete all automated verification, then pause for manual confirmation.

---

## Testing Strategy

### Unit Tests:

- **Parallel Execution**: Test independent node parallel execution, dependency order maintenance, parallel efficiency calculations
- **Feedback System**: Test feedback collection, adaptive replanning triggers, cycle detection, replan attempt limits
- **Progress Callbacks**: Test event emission timing, progress percentage accuracy, callback error handling
- **Visualization**: Test ASCII/Mermaid diagram generation, status indicator accuracy, tree structure rendering
- **Persistence**: Test checkpoint serialization/deserialization, resume functionality, checkpoint listing/deletion

### Integration Tests:

- **End-to-End Parallel Workflow**: Create complex workflow with dependencies, verify parallel groups execute correctly
- **Adaptive Execution Flow**: Create failing workflow, verify feedback triggers replanning, verify eventual success
- **Progress Visualization**: Execute workflow with progress callbacks, generate real-time visualization updates
- **Long-Running Resume**: Create workflow checkpoint, simulate interruption, resume and complete successfully

### Manual Testing Steps:

1. **Create workflow with 5 independent nodes**: Execute and verify completion time is closer to max individual duration than sum
2. **Create workflow with intentional failures**: Verify adaptive replanning occurs and provides meaningful error recovery
3. **Execute workflow with progress visualization**: Verify real-time progress updates and final diagram accuracy
4. **Interrupt and resume long workflow**: Start workflow, save checkpoint, kill process, restart and resume successfully

## Migration Notes

**Backward Compatibility**: All existing `MetaPlanner.execute()` and `WorkflowExecutor.execute_plan()` methods maintain full backward compatibility. New features are opt-in through additional parameters or new methods.

**Configuration Migration**: Users can gradually adopt features:
- Phase 1: No changes needed, automatic parallel execution benefit
- Phase 2: Opt-in to adaptive execution with `execute_with_adaptation()`
- Phase 3: Opt-in to progress callbacks via constructor parameter
- Phase 4: Opt-in to persistence with `enable_persistence=True`

**Data Format Compatibility**: ExecutionResult and WorkflowNode structures are extended, not changed. Existing serialization/deserialization code will continue to work.

## References

- Research: `C:\Users\strau\clawd\vibe-aigc\.wreckit\items\002-advanced-execution-feedback\research.md`
- Current executor: `vibe_aigc/executor.py:95-96` (sequential execution)
- Dependency management: `vibe_aigc/executor.py:162-169` (_dependencies_satisfied method)
- Integration point: `vibe_aigc/planner.py:97-102` (plan_and_execute method)
- Error handling patterns: `tests/test_error_handling.py` (comprehensive error handling examples)
- Vibe AIGC paper: [arXiv:2602.04575](https://arxiv.org/abs/2602.04575) (feedback loop architecture)