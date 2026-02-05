"""MetaPlanner: Core orchestration and Vibe decomposition."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .llm import LLMClient, LLMConfig
from .executor import WorkflowExecutor, ExecutionResult, ExecutionStatus, ProgressEvent
from .visualization import WorkflowVisualizer, VisualizationFormat


class MetaPlanner:
    """Central system architect that decomposes Vibes into executable workflows with adaptive capabilities."""

    def __init__(self, llm_config: Optional[LLMConfig] = None, progress_callback: Optional[Callable[[ProgressEvent], None]] = None):
        self.llm_client = LLMClient(llm_config)
        self.executor = WorkflowExecutor(progress_callback)
        self.max_replan_attempts = 3  # New configuration
        self.replan_history: List[Dict[str, Any]] = []  # New field for tracking adaptation history
        self.progress_callback = progress_callback

    async def plan(self, vibe: Vibe) -> WorkflowPlan:
        """Generate a WorkflowPlan from a Vibe using LLM decomposition."""

        try:
            # Get structured decomposition from LLM
            plan_data = await self.llm_client.decompose_vibe(vibe)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate workflow plan for vibe '{vibe.description}': {e}"
            ) from e

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

        try:
            # Generate execution plan
            plan = await self.plan(vibe)
        except Exception as e:
            raise RuntimeError(
                f"Failed to plan workflow for vibe '{vibe.description}': {e}"
            ) from e

        try:
            # Execute the plan
            execution_result = await self.executor.execute_plan(plan)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute workflow plan '{plan.id}': {e}"
            ) from e

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
                    "result_summary": execution_result.get_summary(),
                    "feedback_data": execution_result.feedback_data,
                    "replan_suggestions": execution_result.replan_suggestions
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
                if execution_result.status == ExecutionStatus.COMPLETED:
                    return self._format_adaptive_result(execution_result, plan, execution_history)
                elif attempt >= self.max_replan_attempts - 1:
                    # Max attempts reached and still failing
                    raise RuntimeError(f"Failed to execute vibe after {self.max_replan_attempts} attempts")
                else:
                    return self._format_adaptive_result(execution_result, plan, execution_history)

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
                {
                    "id": node_id,
                    "error": result.error,
                    "type": self._get_node_type_from_plan(failed_plan, node_id),
                    "quality": execution_result.feedback_data.get(node_id, {}).get("execution_quality", 0.0)
                }
                for node_id, result in execution_result.node_results.items()
                if result.status == ExecutionStatus.FAILED
            ],
            "feedback_data": execution_result.feedback_data,
            "suggestions": execution_result.replan_suggestions,
            "overall_quality": self._calculate_overall_quality(execution_result)
        }

        # Generate adaptation based on analysis
        adaptation = self._generate_adaptation_strategy(original_vibe, failure_context)

        # Record adaptation in history
        self.replan_history.append({
            "timestamp": datetime.now().isoformat(),
            "original_description": original_vibe.description,
            "failure_context": failure_context,
            "adaptation_strategy": adaptation
        })

        return Vibe(
            description=adaptation.get("adapted_description", original_vibe.description),
            style=original_vibe.style,
            constraints=adaptation.get("adapted_constraints", original_vibe.constraints),
            domain=original_vibe.domain,
            metadata={
                **original_vibe.metadata,
                "adaptation_reason": adaptation.get("reason"),
                "original_description": original_vibe.description,
                "adaptation_attempt": len(self.replan_history)
            }
        )

    async def _adapt_vibe_from_error(self, original_vibe: Vibe, error: str) -> Vibe:
        """Generate adapted vibe based on unexpected error."""

        # Simple error-based adaptation
        adapted_description = f"{original_vibe.description} (simplified approach due to error: {error[:50]}...)"
        adapted_constraints = original_vibe.constraints + ["use simpler approach", "avoid complex operations"]

        return Vibe(
            description=adapted_description,
            style=original_vibe.style,
            constraints=adapted_constraints,
            domain=original_vibe.domain,
            metadata={
                **original_vibe.metadata,
                "error_adaptation": True,
                "original_error": error,
                "original_description": original_vibe.description
            }
        )

    def _get_node_type_from_plan(self, plan: WorkflowPlan, node_id: str) -> str:
        """Get node type from plan by node ID."""
        def search_nodes(nodes: List[WorkflowNode]) -> Optional[str]:
            for node in nodes:
                if node.id == node_id:
                    return node.type.value
                # Search children recursively
                child_type = search_nodes(node.children)
                if child_type:
                    return child_type
            return None

        return search_nodes(plan.root_nodes) or "unknown"

    def _calculate_overall_quality(self, execution_result: ExecutionResult) -> float:
        """Calculate overall execution quality from feedback data."""
        if not execution_result.feedback_data:
            return 0.0

        qualities = [
            feedback.get("execution_quality", 0.0)
            for feedback in execution_result.feedback_data.values()
        ]

        return sum(qualities) / len(qualities) if qualities else 0.0

    def _generate_adaptation_strategy(self, original_vibe: Vibe, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptation strategy based on failure analysis."""

        failed_nodes = failure_context["failed_nodes"]
        overall_quality = failure_context["overall_quality"]
        suggestions = failure_context["suggestions"]

        # Analyze failure patterns
        error_types = [node.get("error", "") for node in failed_nodes]
        common_issues = []

        # Pattern detection
        if any("timeout" in error.lower() for error in error_types):
            common_issues.append("timeout_issues")
        if any("memory" in error.lower() for error in error_types):
            common_issues.append("memory_issues")
        if overall_quality < 0.3:
            common_issues.append("quality_issues")

        # Generate adaptation
        adapted_description = original_vibe.description
        adapted_constraints = list(original_vibe.constraints)

        if "timeout_issues" in common_issues:
            adapted_description += " (with shorter, simpler tasks)"
            adapted_constraints.append("break complex tasks into smaller steps")
            adapted_constraints.append("use faster execution methods")

        if "memory_issues" in common_issues:
            adapted_constraints.append("minimize memory usage")
            adapted_constraints.append("use streaming approaches where possible")

        if "quality_issues" in common_issues:
            adapted_description += " (with focus on quality and detail)"
            adapted_constraints.append("prioritize output quality over speed")

        # Include suggestions from feedback
        for suggestion in suggestions:
            if "suggested_changes" in suggestion.get("suggestion", suggestion):
                for change in suggestion.get("suggestion", suggestion).get("suggested_changes", []):
                    if change not in adapted_constraints:
                        adapted_constraints.append(change)

        return {
            "adapted_description": adapted_description,
            "adapted_constraints": adapted_constraints,
            "reason": f"Adapted due to: {', '.join(common_issues)}",
            "failure_patterns": common_issues
        }

    def _format_adaptive_result(self, execution_result: ExecutionResult,
                              plan: WorkflowPlan, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format result with adaptation information."""

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
            "adaptation_info": {
                "total_attempts": len(execution_history),
                "adaptation_history": execution_history,
                "replan_history": self.replan_history,
                "final_feedback": execution_result.feedback_data,
                "final_suggestions": execution_result.replan_suggestions
            }
        }

        return base_result

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
