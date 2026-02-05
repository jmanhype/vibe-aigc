"""Workflow execution engine for running WorkflowPlans."""

import asyncio
from typing import Any, Dict, List, Set
from datetime import datetime
from enum import Enum

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
                 result: Any = None, error: str = None, duration: float = 0.0):
        self.node_id = node_id
        self.status = status
        self.result = result
        self.error = error
        self.duration = duration
        self.started_at = datetime.now().isoformat()


class ExecutionResult:
    """Complete result of WorkflowPlan execution."""

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.status = ExecutionStatus.PENDING
        self.node_results: Dict[str, NodeResult] = {}
        self.started_at = datetime.now().isoformat()
        self.completed_at: str = None
        self.total_duration: float = 0.0

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
            "completed_at": self.completed_at
        }


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
        """Execute a complete WorkflowPlan."""

        result = ExecutionResult(plan.id)
        result.status = ExecutionStatus.RUNNING

        execution_failed = False

        try:
            # Execute root nodes (for now, sequentially)
            for node in plan.root_nodes:
                try:
                    await self._execute_node_tree(node, result)
                except Exception as e:
                    execution_failed = True
                    # Mark this tree as failed, but continue with other roots
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

        # Calculate total duration
        result.total_duration = sum(r.duration for r in result.node_results.values())

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

            # Execute children sequentially
            for child in node.children:
                await self._execute_node_tree(child, result)

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