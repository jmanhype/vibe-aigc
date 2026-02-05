"""MetaPlanner: Core orchestration and Vibe decomposition."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .llm import LLMClient, LLMConfig
from .executor import WorkflowExecutor, ExecutionResult


class MetaPlanner:
    """Central system architect that decomposes Vibes into executable workflows."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(llm_config)
        self.executor = WorkflowExecutor()

    async def plan(self, vibe: Vibe) -> WorkflowPlan:
        """Generate a WorkflowPlan from a Vibe using LLM decomposition."""

        # Get structured decomposition from LLM
        plan_data = await self.llm_client.decompose_vibe(vibe)

        # Convert LLM response to structured WorkflowPlan
        plan_id = plan_data.get("id", f"plan-{uuid.uuid4().hex[:8]}")

        root_nodes = [
            self._build_workflow_node(node_data)
            for node_data in plan_data.get("root_nodes", [])
        ]

        # Calculate total estimated duration
        total_duration = sum(
            self._calculate_node_duration(node)
            for node in root_nodes
        )

        return WorkflowPlan(
            id=plan_id,
            source_vibe=vibe,
            root_nodes=root_nodes,
            estimated_total_duration=total_duration,
            created_at=datetime.now().isoformat()
        )

    async def execute(self, vibe: Vibe) -> Dict[str, Any]:
        """Plan and execute a Vibe workflow with full execution engine."""

        # Generate execution plan
        plan = await self.plan(vibe)

        # Execute the plan
        execution_result = await self.executor.execute_plan(plan)

        # Format result for API compatibility
        summary = execution_result.get_summary()

        return {
            "status": summary["status"],
            "plan_id": summary["plan_id"],
            "vibe_description": vibe.description,
            "execution_summary": {
                "total_nodes": summary["total_nodes"],
                "completed": summary["completed"],
                "failed": summary["failed"],
                "total_duration": summary["total_duration"],
                "started_at": summary["started_at"],
                "completed_at": summary["completed_at"]
            },
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

    async def plan_and_execute(self, vibe: Vibe) -> tuple[WorkflowPlan, ExecutionResult]:
        """Get both the plan and execution result (for detailed analysis)."""

        plan = await self.plan(vibe)
        execution_result = await self.executor.execute_plan(plan)
        return plan, execution_result

    def _build_workflow_node(self, node_data: Dict[str, Any]) -> WorkflowNode:
        """Convert LLM response data to WorkflowNode structure."""

        # Validate and normalize node type
        node_type_str = node_data.get("type", "generate").lower()
        try:
            node_type = WorkflowNodeType(node_type_str)
        except ValueError:
            node_type = WorkflowNodeType.GENERATE

        # Recursively build children nodes
        children = [
            self._build_workflow_node(child_data)
            for child_data in node_data.get("children", [])
        ]

        return WorkflowNode(
            id=node_data.get("id", f"task-{uuid.uuid4().hex[:8]}"),
            type=node_type,
            description=node_data.get("description", "Untitled task"),
            parameters=node_data.get("parameters", {}),
            dependencies=node_data.get("dependencies", []),
            children=children,
            estimated_duration=node_data.get("estimated_duration")
        )

    def _calculate_node_duration(self, node: WorkflowNode) -> int:
        """Calculate total estimated duration for a node and its children."""

        node_duration = node.estimated_duration or 0
        children_duration = sum(
            self._calculate_node_duration(child)
            for child in node.children
        )

        return node_duration + children_duration