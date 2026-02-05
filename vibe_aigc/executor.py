"""Workflow execution engine for running WorkflowPlans."""

import asyncio
from typing import Any, Dict, List, Set, Optional, Callable, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import time

from .models import WorkflowPlan, WorkflowNode, WorkflowNodeType

if TYPE_CHECKING:
    from .persistence import WorkflowCheckpoint


class ExecutionStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


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


class NodeResult:
    """Result of executing a single WorkflowNode."""

    def __init__(self, node_id: str, status: ExecutionStatus,
                 result: Any = None, error: Optional[str] = None, duration: float = 0.0):
        self.node_id = node_id
        self.status = status
        self.result = result
        self.error = error
        self.duration = duration
        self.started_at = datetime.now().isoformat()


class ExecutionResult:
    """Complete result of WorkflowPlan execution with parallel tracking."""

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.status = ExecutionStatus.PENDING
        self.node_results: Dict[str, NodeResult] = {}
        self.started_at = datetime.now().isoformat()
        self.completed_at: Optional[str] = None
        self.total_duration: float = 0.0
        self.parallel_efficiency: float = 0.0  # New field for tracking parallel execution benefits
        self.execution_groups: List[List[str]] = []  # New field for tracking parallel execution groups
        self.feedback_data: Dict[str, Any] = {}  # New field for storing node execution feedback
        self.replan_suggestions: List[Dict[str, Any]] = []  # New field for storing replanning suggestions

    def add_node_result(self, result: NodeResult):
        """Add result for a completed node."""
        self.node_results[result.node_id] = result

    def is_complete(self) -> bool:
        """Check if all nodes have completed (successfully or failed)."""
        return all(
            result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
            for result in self.node_results.values()
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        completed = sum(1 for r in self.node_results.values() if r.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for r in self.node_results.values() if r.status == ExecutionStatus.FAILED)
        total = len(self.node_results)

        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "total_nodes": total,
            "completed": completed,
            "failed": failed,
            "total_duration": self.total_duration,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "parallel_efficiency": self.parallel_efficiency,
            "execution_groups": len(self.execution_groups)
        }

    def calculate_parallel_efficiency(self) -> float:
        """Calculate efficiency gained from parallel execution."""
        if not self.node_results:
            return 0.0

        total_node_duration = sum(r.duration for r in self.node_results.values())
        actual_duration = self.total_duration

        return max(0.0, (total_node_duration - actual_duration) / total_node_duration) if total_node_duration > 0 else 0.0

    def add_feedback(self, node_id: str, feedback: Dict[str, Any]):
        """Add feedback data from node execution."""
        self.feedback_data[node_id] = feedback

    def suggest_replan(self, suggestion: Dict[str, Any]):
        """Add replanning suggestion based on execution results."""
        self.replan_suggestions.append({
            "timestamp": datetime.now().isoformat(),
            **suggestion  # Flatten the suggestion into the main dict
        })

    def should_replan(self) -> bool:
        """Determine if replanning is recommended based on execution results."""
        # Check for multiple failures, resource constraints, or explicit suggestions
        failed_nodes = sum(1 for r in self.node_results.values() if r.status == ExecutionStatus.FAILED)
        return failed_nodes > 1 or len(self.replan_suggestions) > 0


class WorkflowExecutor:
    """Basic execution engine for WorkflowPlans with progress callbacks."""

    def __init__(self, progress_callback: Optional[Callable[[ProgressEvent], None]] = None):
        self.node_handlers = {
            WorkflowNodeType.ANALYZE: self._execute_analyze,
            WorkflowNodeType.GENERATE: self._execute_generate,
            WorkflowNodeType.TRANSFORM: self._execute_transform,
            WorkflowNodeType.VALIDATE: self._execute_validate,
            WorkflowNodeType.COMPOSITE: self._execute_composite
        }
        self.progress_callback = progress_callback

    async def execute_plan(self, plan: WorkflowPlan,
                          resume_from_checkpoint: Optional['WorkflowCheckpoint'] = None) -> ExecutionResult:
        """Execute a complete WorkflowPlan with parallel execution and progress tracking.

        Args:
            plan: The WorkflowPlan to execute
            resume_from_checkpoint: Optional checkpoint to resume from. If provided,
                                   already-completed nodes will be skipped.

        Returns:
            ExecutionResult with combined results from checkpoint and new execution
        """
        # Initialize result - either from checkpoint or fresh
        if resume_from_checkpoint:
            result = resume_from_checkpoint.execution_result
            # Reset status to running for resumed execution
            result.status = ExecutionStatus.RUNNING
            completed_node_ids = self._get_completed_node_ids(result)
            self._emit_progress(ProgressEventType.WORKFLOW_STARTED,
                              message=f"Resuming workflow: {plan.id} ({len(completed_node_ids)} nodes already complete)")
        else:
            result = ExecutionResult(plan.id)
            result.status = ExecutionStatus.RUNNING
            completed_node_ids = set()
            self._emit_progress(ProgressEventType.WORKFLOW_STARTED,
                              message=f"Starting workflow: {plan.id}")

        start_time = time.time()
        execution_failed = False

        try:
            # Identify remaining nodes (exclude already completed when resuming)
            remaining_root_nodes = self._identify_remaining_nodes(plan.root_nodes, completed_node_ids)

            # Group independent nodes for parallel execution
            parallel_groups = self._build_parallel_execution_groups(remaining_root_nodes, result)
            result.execution_groups = [[node.id for node in group] for group in parallel_groups]

            # Calculate progress tracking
            total_nodes = self._count_total_nodes(plan.root_nodes)
            completed_nodes = len(completed_node_ids)  # Start from already completed count

            # Execute each parallel group in sequence, nodes within groups in parallel
            for group_idx, group in enumerate(parallel_groups):
                self._emit_progress(ProgressEventType.GROUP_STARTED,
                                  message=f"Starting parallel group {group_idx + 1}/{len(parallel_groups)}",
                                  metadata={"group_size": len(group), "group_nodes": [n.id for n in group]})

                group_tasks = [
                    self._execute_node_tree_with_progress(node, result, total_nodes, completed_nodes)
                    for node in group
                ]
                try:
                    await asyncio.gather(*group_tasks, return_exceptions=True)
                    completed_nodes += len(group)

                    progress_percent = (completed_nodes / total_nodes) * 100 if total_nodes > 0 else 100
                    self._emit_progress(ProgressEventType.GROUP_COMPLETED,
                                      progress_percent=progress_percent,
                                      message=f"Completed group {group_idx + 1}/{len(parallel_groups)}")

                except Exception as e:
                    execution_failed = True
                    # Mark failed nodes, continue with remaining groups
                    for node in group:
                        if node.id not in result.node_results:
                            self._mark_tree_failed(node, result, str(e))

            # Determine final status based on whether any nodes failed
            if execution_failed or any(r.status == ExecutionStatus.FAILED for r in result.node_results.values()):
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

        except Exception as e:
            # Unexpected error during execution setup
            result.status = ExecutionStatus.FAILED
            result.completed_at = datetime.now().isoformat()

            self._emit_progress(ProgressEventType.WORKFLOW_COMPLETED,
                              progress_percent=100.0,
                              message=f"Workflow failed with error: {str(e)}")

            # Add error result for any nodes that haven't been executed
            for node in plan.root_nodes:
                self._mark_tree_failed(node, result, str(e))

        # Calculate durations and parallel efficiency
        result.total_duration = time.time() - start_time
        result.parallel_efficiency = result.calculate_parallel_efficiency()

        return result


    def _dependencies_satisfied(self, node: WorkflowNode, result: ExecutionResult) -> bool:
        """Check if all dependencies for a node have completed successfully."""

        for dep_id in node.dependencies:
            dep_result = result.node_results.get(dep_id)
            if not dep_result or dep_result.status != ExecutionStatus.COMPLETED:
                return False
        return True

    def _get_completed_node_ids(self, result: ExecutionResult) -> Set[str]:
        """Get set of node IDs that have already completed successfully."""
        return {
            node_id for node_id, node_result in result.node_results.items()
            if node_result.status == ExecutionStatus.COMPLETED
        }

    def _identify_remaining_nodes(self, nodes: List[WorkflowNode],
                                  completed_node_ids: Set[str]) -> List[WorkflowNode]:
        """Identify nodes that still need to be executed.

        Returns a list of nodes (with filtered children) that need execution.
        Completed nodes are excluded, but their incomplete children are retained
        under a synthetic parent structure.

        Args:
            nodes: Root nodes to filter
            completed_node_ids: Set of already completed node IDs

        Returns:
            List of nodes needing execution
        """
        remaining = []

        for node in nodes:
            if node.id in completed_node_ids:
                # Node is done, but check if children need execution
                remaining_children = self._identify_remaining_nodes(node.children, completed_node_ids)
                # Add remaining children as independent nodes (parent completed)
                remaining.extend(remaining_children)
            else:
                # Node needs execution - filter its children too
                filtered_node = WorkflowNode(
                    id=node.id,
                    type=node.type,
                    description=node.description,
                    parameters=node.parameters,
                    dependencies=node.dependencies,
                    children=self._identify_remaining_nodes(node.children, completed_node_ids),
                    estimated_duration=node.estimated_duration
                )
                remaining.append(filtered_node)

        return remaining

    def _mark_tree_failed(self, node: WorkflowNode, result: ExecutionResult, error: str):
        """Mark a node and its children as failed."""

        if node.id not in result.node_results:
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.FAILED, error=error
            ))

        for child in node.children:
            self._mark_tree_failed(child, result, error)

    def _build_parallel_execution_groups(self, nodes: List[WorkflowNode], result: ExecutionResult) -> List[List[WorkflowNode]]:
        """Group nodes into parallel execution batches based on dependencies.

        Uses topological sorting to identify parallelizable groups where each group
        can execute in parallel, but groups must execute in sequence.
        """
        if not nodes:
            return []

        # Build dependency graph - use node IDs instead of nodes for hashability
        node_map = {node.id: node for node in nodes}
        groups = []
        remaining_node_ids = {node.id for node in nodes}
        processed_nodes = set()

        # Add already completed nodes to processed set
        for node_id in result.node_results:
            if result.node_results[node_id].status == ExecutionStatus.COMPLETED:
                processed_nodes.add(node_id)

        while remaining_node_ids:
            # Find nodes with no unprocessed dependencies
            ready_node_ids = []
            for node_id in remaining_node_ids:
                node = node_map[node_id]
                # Check if all dependencies are either processed or not in our scope
                dependencies_satisfied = all(
                    dep_id in processed_nodes or dep_id not in node_map
                    for dep_id in node.dependencies
                )
                if dependencies_satisfied:
                    ready_node_ids.append(node_id)

            if not ready_node_ids:
                # This shouldn't happen with valid dependency graphs, but handle gracefully
                # Break circular dependencies by taking the first remaining node
                ready_node_ids = [next(iter(remaining_node_ids))]

            # Convert node IDs back to nodes for this group
            ready_nodes = [node_map[node_id] for node_id in ready_node_ids]
            groups.append(ready_nodes)

            # Mark these nodes as ready to be processed
            for node_id in ready_node_ids:
                remaining_node_ids.remove(node_id)
                processed_nodes.add(node_id)

        return groups

    def _count_total_nodes(self, nodes: List[WorkflowNode]) -> int:
        """Count total number of nodes in the workflow tree."""
        total = 0
        for node in nodes:
            total += 1  # Count this node
            total += self._count_total_nodes(node.children)  # Count children recursively
        return total

    async def _execute_node_tree_with_progress(self, node: WorkflowNode,
                                             result: ExecutionResult,
                                             total_nodes: int,
                                             completed_base: int):
        """Execute node tree with progress reporting."""

        # Skip already completed nodes (for resume scenarios)
        if node.id in result.node_results:
            existing = result.node_results[node.id]
            if existing.status == ExecutionStatus.COMPLETED:
                # Node already done, just process children
                if node.children:
                    child_groups = self._build_parallel_execution_groups(node.children, result)
                    for group in child_groups:
                        child_tasks = [
                            self._execute_node_tree_with_progress(child, result, total_nodes, completed_base)
                            for child in group
                        ]
                        await asyncio.gather(*child_tasks, return_exceptions=True)
                return

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
            # Execute the node
            handler = self.node_handlers.get(node.type, self._execute_default)
            node_result = await handler(node)

            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.COMPLETED,
                result=node_result, duration=duration
            ))

            # Emit node completed
            self._emit_progress(ProgressEventType.NODE_COMPLETED,
                              node_id=node.id,
                              message=f"Completed: {node.description}",
                              metadata={"duration": duration, "result_preview": str(node_result)[:100]})

            # Collect execution feedback
            feedback = self._analyze_node_execution(node, node_result, duration)
            result.add_feedback(node.id, feedback)

            # Check for replanning triggers
            if self._should_suggest_replan(node, node_result, feedback):
                result.suggest_replan({
                    "node_id": node.id,
                    "reason": feedback.get("replan_reason", "low_quality_output"),
                    "suggested_changes": feedback.get("optimization_suggestions", [])
                })

            # Execute children with progress tracking
            if node.children:
                child_groups = self._build_parallel_execution_groups(node.children, result)
                for group in child_groups:
                    child_tasks = [
                        self._execute_node_tree_with_progress(child, result, total_nodes, completed_base)
                        for child in group
                    ]
                    await asyncio.gather(*child_tasks, return_exceptions=True)

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.FAILED,
                error=str(e), duration=duration
            ))

            # Emit node failed
            self._emit_progress(ProgressEventType.NODE_FAILED,
                              node_id=node.id,
                              message=f"Failed: {node.description} - {str(e)}",
                              metadata={"duration": duration, "error": str(e)})

            # Enhanced error feedback
            error_feedback = self._analyze_execution_error(node, e, duration)
            result.add_feedback(node.id, error_feedback)

            if error_feedback.get("replannable", False):
                result.suggest_replan({
                    "node_id": node.id,
                    "error": str(e),
                    "replan_reason": "execution_failure",
                    "suggested_changes": error_feedback.get("suggested_changes", [])
                })

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

    def _analyze_node_execution(self, node: WorkflowNode, node_result: Any, duration: float) -> Dict[str, Any]:
        """Analyze node execution result for feedback."""

        # Assess execution quality based on result content and performance
        execution_quality = self._assess_result_quality(node_result)
        resource_usage = self._measure_resource_usage(node, duration)

        feedback = {
            "execution_quality": execution_quality,
            "resource_usage": resource_usage,
            "performance_metrics": {
                "duration": duration,
                "expected_duration": node.estimated_duration,
                "efficiency": self._calculate_node_efficiency(duration, node.estimated_duration)
            },
            "replan_indicators": self._check_replan_indicators(node, node_result, execution_quality),
            "optimization_suggestions": self._suggest_optimizations(node, node_result, duration)
        }

        return feedback

    def _analyze_execution_error(self, node: WorkflowNode, error: Exception, duration: float) -> Dict[str, Any]:
        """Analyze execution error for feedback and replanning suggestions."""

        error_type = type(error).__name__
        error_message = str(error)

        # Categorize errors for replanning decisions
        replannable_errors = ["TimeoutError", "ConnectionError", "HTTPError", "MemoryError", "RuntimeError"]
        is_replannable = error_type in replannable_errors or "timeout" in error_message.lower()

        feedback = {
            "error_type": error_type,
            "error_message": error_message,
            "replannable": is_replannable,
            "execution_quality": 0.0,  # Failed execution
            "resource_usage": self._measure_resource_usage(node, duration),
            "performance_metrics": {
                "duration": duration,
                "expected_duration": node.estimated_duration,
                "failure_point": "execution"
            },
            "suggested_changes": self._suggest_error_recovery(node, error),
            "replan_reason": f"execution_failure_{error_type.lower()}"
        }

        return feedback

    def _assess_result_quality(self, result: Any) -> float:
        """Assess the quality of a node execution result."""
        if result is None:
            return 0.0

        # Basic quality assessment based on result structure
        if isinstance(result, dict):
            # Check for expected fields
            quality = 0.5  # Base quality for dict results

            # Increase quality for meaningful content
            if result.get("result") and len(str(result.get("result", ""))) > 10:
                quality += 0.3

            # Check for error indicators
            if result.get("error") or result.get("failed"):
                quality = max(0.1, quality - 0.4)

            return min(1.0, quality)

        # Non-dict results get moderate quality
        return 0.7 if result else 0.0

    def _measure_resource_usage(self, node: WorkflowNode, duration: float) -> Dict[str, Any]:
        """Measure resource usage for a node execution."""
        return {
            "cpu_time": duration,  # Simplified - could use psutil for real CPU metrics
            "memory_usage": "normal",  # Placeholder - could implement real memory tracking
            "efficiency_ratio": self._calculate_node_efficiency(duration, node.estimated_duration)
        }

    def _calculate_node_efficiency(self, actual_duration: float, estimated_duration: Optional[int]) -> float:
        """Calculate efficiency ratio of actual vs estimated duration."""
        if not estimated_duration or estimated_duration <= 0:
            return 1.0  # No estimate available

        # Efficiency is inverse of duration ratio - lower actual time = higher efficiency
        return min(2.0, estimated_duration / max(0.1, actual_duration))

    def _check_replan_indicators(self, node: WorkflowNode, result: Any, quality: float) -> List[str]:
        """Check for indicators that suggest replanning might be beneficial."""
        indicators = []

        # Quality-based indicators
        if quality < 0.3:
            indicators.append("low_quality_output")

        # Result content indicators
        if isinstance(result, dict):
            if result.get("error"):
                indicators.append("execution_error")
            if result.get("timeout"):
                indicators.append("performance_issue")

        return indicators

    def _suggest_optimizations(self, node: WorkflowNode, result: Any, duration: float) -> List[str]:
        """Suggest optimizations based on execution analysis."""
        suggestions = []

        # Performance-based suggestions - suggest if actual duration exceeds expected
        if node.estimated_duration and node.estimated_duration > 0:
            if duration > float(node.estimated_duration) * 1.5:  # 50% slower than expected
                suggestions.append("reduce_task_complexity")
                suggestions.append("parallelize_subtasks")
        elif node.estimated_duration == 0 and duration > 0.1:  # For tasks with no estimate, suggest if > 0.1s
            suggestions.append("reduce_task_complexity")

        # Quality-based suggestions
        if isinstance(result, dict) and len(str(result.get("result", ""))) < 20:
            suggestions.append("enhance_output_detail")

        return suggestions

    def _suggest_error_recovery(self, node: WorkflowNode, error: Exception) -> List[str]:
        """Suggest recovery strategies based on error type."""
        error_type = type(error).__name__
        suggestions = []

        if "timeout" in str(error).lower() or error_type == "TimeoutError":
            suggestions.extend(["increase_timeout", "break_into_smaller_tasks"])
        elif "memory" in str(error).lower() or error_type == "MemoryError":
            suggestions.extend(["reduce_data_size", "use_streaming_approach"])
        elif "connection" in str(error).lower() or error_type in ["ConnectionError", "HTTPError"]:
            suggestions.extend(["retry_with_backoff", "use_alternative_endpoint"])
        else:
            suggestions.append("retry_with_different_parameters")

        return suggestions

    def _should_suggest_replan(self, node: WorkflowNode, result: Any, feedback: Dict[str, Any]) -> bool:
        """Determine if execution suggests replanning."""
        quality_threshold = 0.7  # Configurable

        # Check quality threshold
        if feedback.get("execution_quality", 1.0) < quality_threshold:
            return True

        # Check for specific replan indicators
        if len(feedback.get("replan_indicators", [])) >= 2:
            return True

        return False

    # Basic node type handlers (mock implementations for Phase 3)

    async def _execute_analyze(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute analysis task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "type": "analysis",
            "description": node.description,
            "result": f"Analysis completed for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_generate(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute generation task."""
        await asyncio.sleep(0.2)  # Simulate work
        return {
            "type": "generation",
            "description": node.description,
            "result": f"Generated content for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_transform(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute transformation task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "type": "transformation",
            "description": node.description,
            "result": f"Transformed content for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_validate(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute validation task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "type": "validation",
            "description": node.description,
            "result": f"Validation passed for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_composite(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute composite task (children handle the work)."""
        return {
            "type": "composite",
            "description": node.description,
            "result": f"Composite task organized: {node.description}",
            "child_count": len(node.children)
        }

    async def _execute_default(self, node: WorkflowNode) -> Dict[str, Any]:
        """Default handler for unknown node types."""
        await asyncio.sleep(0.1)
        return {
            "type": "default",
            "description": node.description,
            "result": f"Default execution for: {node.description}"
        }