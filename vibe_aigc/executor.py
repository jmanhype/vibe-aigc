"""Workflow execution engine for running WorkflowPlans."""

import asyncio
from typing import Any, Dict, List, Set, Optional
from datetime import datetime
from enum import Enum
import time

from .models import WorkflowPlan, WorkflowNode, WorkflowNodeType


class ExecutionStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


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


class WorkflowExecutor:
    """Basic execution engine for WorkflowPlans."""

    def __init__(self):
        self.node_handlers = {
            WorkflowNodeType.ANALYZE: self._execute_analyze,
            WorkflowNodeType.GENERATE: self._execute_generate,
            WorkflowNodeType.TRANSFORM: self._execute_transform,
            WorkflowNodeType.VALIDATE: self._execute_validate,
            WorkflowNodeType.COMPOSITE: self._execute_composite
        }

    async def execute_plan(self, plan: WorkflowPlan) -> ExecutionResult:
        """Execute a complete WorkflowPlan with parallel execution of independent nodes."""

        result = ExecutionResult(plan.id)
        result.status = ExecutionStatus.RUNNING

        start_time = time.time()
        execution_failed = False

        try:
            # Group independent root nodes for parallel execution
            parallel_groups = self._build_parallel_execution_groups(plan.root_nodes, result)
            result.execution_groups = [[node.id for node in group] for group in parallel_groups]

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

            # Determine final status based on whether any nodes failed
            if execution_failed or any(r.status == ExecutionStatus.FAILED for r in result.node_results.values()):
                result.status = ExecutionStatus.FAILED
            else:
                result.status = ExecutionStatus.COMPLETED

            result.completed_at = datetime.now().isoformat()

        except Exception as e:
            # Unexpected error during execution setup
            result.status = ExecutionStatus.FAILED
            result.completed_at = datetime.now().isoformat()

            # Add error result for any nodes that haven't been executed
            for node in plan.root_nodes:
                self._mark_tree_failed(node, result, str(e))

        # Calculate durations and parallel efficiency
        result.total_duration = time.time() - start_time
        result.parallel_efficiency = result.calculate_parallel_efficiency()

        return result

    async def _execute_node_tree(self, node: WorkflowNode, result: ExecutionResult):
        """Execute a node and its children, respecting dependencies."""

        # Check if dependencies are satisfied
        if not self._dependencies_satisfied(node, result):
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.SKIPPED,
                error="Dependencies not satisfied"
            ))
            return

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

            # Execute children in parallel groups
            if node.children:
                child_groups = self._build_parallel_execution_groups(node.children, result)
                for group in child_groups:
                    child_tasks = [
                        self._execute_node_tree(child, result)
                        for child in group
                    ]
                    await asyncio.gather(*child_tasks, return_exceptions=True)

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.FAILED,
                error=str(e), duration=duration
            ))
            raise

    def _dependencies_satisfied(self, node: WorkflowNode, result: ExecutionResult) -> bool:
        """Check if all dependencies for a node have completed successfully."""

        for dep_id in node.dependencies:
            dep_result = result.node_results.get(dep_id)
            if not dep_result or dep_result.status != ExecutionStatus.COMPLETED:
                return False
        return True

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