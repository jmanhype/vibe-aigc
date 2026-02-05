"""Workflow visualization utilities."""

from typing import List, Dict, Set, Optional
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
            # Calculate actual total duration from node results
            total_duration = sum(result.duration for result in execution_result.node_results.values() if result.duration)
            lines.append(f"Duration: {total_duration:.2f}s")
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
        node_line = f"{prefix}{connector}{status_indicator} [{node.type.value}] {node.description}"
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

            # Node shape based on type and status
            shape = {
                "analyze": "([{}])",
                "generate": "[{}]",
                "transform": "{{{{{}}}}}",
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