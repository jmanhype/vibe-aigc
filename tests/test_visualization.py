"""Test workflow visualization functionality."""

import pytest

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import ExecutionResult, NodeResult, ExecutionStatus
from vibe_aigc.visualization import WorkflowVisualizer, VisualizationFormat

class TestWorkflowVisualization:
    """Test workflow diagram generation functionality."""

    def test_ascii_diagram_generation_basic(self):
        """Test basic ASCII workflow diagram generation."""

        # Create simple workflow
        node = WorkflowNode(
            id="test-node",
            type=WorkflowNodeType.GENERATE,
            description="Test node"
        )

        plan = WorkflowPlan(
            id="test-viz",
            source_vibe=Vibe(description="Test visualization"),
            root_nodes=[node]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII)

        # Verify diagram content
        assert "Workflow Plan: test-viz" in diagram
        assert "Source Vibe: Test visualization" in diagram
        assert "Test node" in diagram
        assert "⏳" in diagram  # Pending status indicator
        assert "[generate]" in diagram  # Node type
        assert "└──" in diagram  # Tree structure

    def test_ascii_diagram_with_hierarchy(self):
        """Test ASCII diagram with hierarchical workflow structure."""

        # Create hierarchical workflow
        child1 = WorkflowNode(
            id="analyze-1",
            type=WorkflowNodeType.ANALYZE,
            description="Analyze input"
        )

        child2 = WorkflowNode(
            id="generate-1",
            type=WorkflowNodeType.GENERATE,
            description="Generate content",
            dependencies=["analyze-1"]
        )

        parent = WorkflowNode(
            id="workflow-1",
            type=WorkflowNodeType.COMPOSITE,
            description="Complete workflow",
            children=[child1, child2]
        )

        plan = WorkflowPlan(
            id="hierarchy-test",
            source_vibe=Vibe(description="Test hierarchy"),
            root_nodes=[parent]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII)

        # Verify hierarchical structure
        assert "Complete workflow" in diagram
        assert "Analyze input" in diagram
        assert "Generate content" in diagram
        assert "├──" in diagram or "└──" in diagram  # Tree structure
        assert "[composite]" in diagram
        assert "[analyze]" in diagram
        assert "[generate]" in diagram
        assert "Depends on: analyze-1" in diagram  # Dependencies shown

    def test_ascii_diagram_with_execution_results(self):
        """Test ASCII diagram showing execution results."""

        # Create workflow
        node1 = WorkflowNode(
            id="completed-node",
            type=WorkflowNodeType.ANALYZE,
            description="Completed task"
        )

        node2 = WorkflowNode(
            id="failed-node",
            type=WorkflowNodeType.GENERATE,
            description="Failed task"
        )

        plan = WorkflowPlan(
            id="result-test",
            source_vibe=Vibe(description="Test results"),
            root_nodes=[node1, node2]
        )

        # Create execution result
        result = ExecutionResult("result-test")
        result.add_node_result(NodeResult("completed-node", ExecutionStatus.COMPLETED, duration=1.5))
        result.add_node_result(NodeResult("failed-node", ExecutionStatus.FAILED, error="Test error", duration=0.8))
        result.status = ExecutionStatus.FAILED

        diagram = WorkflowVisualizer.generate_diagram(plan, result, VisualizationFormat.ASCII)

        # Verify status indicators
        assert "✅" in diagram  # Completed node
        assert "❌" in diagram  # Failed node
        assert "Completed task" in diagram
        assert "Failed task" in diagram

        # Verify execution summary
        assert "Execution Summary:" in diagram
        assert "Status: failed" in diagram
        assert "Duration: 2.30s" in diagram  # Sum of durations

    def test_ascii_diagram_with_dependencies(self):
        """Test ASCII diagram correctly shows dependencies."""

        # Create workflow with dependencies
        step1 = WorkflowNode(
            id="step-1",
            type=WorkflowNodeType.ANALYZE,
            description="First step"
        )

        step2 = WorkflowNode(
            id="step-2",
            type=WorkflowNodeType.GENERATE,
            description="Second step",
            dependencies=["step-1"]
        )

        step3 = WorkflowNode(
            id="step-3",
            type=WorkflowNodeType.VALIDATE,
            description="Third step",
            dependencies=["step-2", "step-1"]
        )

        plan = WorkflowPlan(
            id="dependency-test",
            source_vibe=Vibe(description="Test dependencies"),
            root_nodes=[step1, step2, step3]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan)

        # Verify dependencies are shown
        assert "Depends on: step-1" in diagram
        assert "Depends on: step-2, step-1" in diagram
        assert "First step" in diagram
        assert "Second step" in diagram
        assert "Third step" in diagram

    def test_ascii_diagram_multiple_root_nodes(self):
        """Test ASCII diagram with multiple root nodes."""

        # Create multiple root workflows
        root1 = WorkflowNode(
            id="root-1",
            type=WorkflowNodeType.ANALYZE,
            description="First root task"
        )

        root2 = WorkflowNode(
            id="root-2",
            type=WorkflowNodeType.GENERATE,
            description="Second root task"
        )

        root3 = WorkflowNode(
            id="root-3",
            type=WorkflowNodeType.VALIDATE,
            description="Third root task"
        )

        plan = WorkflowPlan(
            id="multi-root-test",
            source_vibe=Vibe(description="Multiple roots"),
            root_nodes=[root1, root2, root3]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan)

        # Verify all root nodes appear
        assert "First root task" in diagram
        assert "Second root task" in diagram
        assert "Third root task" in diagram

        # Should have both ├── and └── connectors for multiple roots
        assert "├──" in diagram  # Non-last roots
        assert "└──" in diagram  # Last root

    def test_ascii_diagram_with_parallel_efficiency(self):
        """Test ASCII diagram includes parallel efficiency when available."""

        node = WorkflowNode(
            id="parallel-node",
            type=WorkflowNodeType.GENERATE,
            description="Parallel test"
        )

        plan = WorkflowPlan(
            id="parallel-test",
            source_vibe=Vibe(description="Test parallel"),
            root_nodes=[node]
        )

        # Create result with parallel efficiency
        result = ExecutionResult("parallel-test")
        result.add_node_result(NodeResult("parallel-node", ExecutionStatus.COMPLETED, duration=2.0))
        result.parallel_efficiency = 0.35  # 35% efficiency gain
        result.status = ExecutionStatus.COMPLETED

        diagram = WorkflowVisualizer.generate_diagram(plan, result)

        # Verify parallel efficiency is shown
        assert "Parallel Efficiency: 35.0%" in diagram

    def test_ascii_diagram_estimated_duration(self):
        """Test ASCII diagram shows estimated duration when available."""

        node = WorkflowNode(
            id="timed-node",
            type=WorkflowNodeType.GENERATE,
            description="Timed task",
            estimated_duration=30
        )

        plan = WorkflowPlan(
            id="duration-test",
            source_vibe=Vibe(description="Test duration"),
            root_nodes=[node],
            estimated_total_duration=30
        )

        diagram = WorkflowVisualizer.generate_diagram(plan)

        # Verify estimated duration is shown
        assert "Estimated Duration: 30s" in diagram

    def test_mermaid_diagram_generation_basic(self):
        """Test basic Mermaid workflow diagram generation."""

        node = WorkflowNode(
            id="test-node",
            type=WorkflowNodeType.GENERATE,
            description="Test node for mermaid"
        )

        plan = WorkflowPlan(
            id="mermaid-test",
            source_vibe=Vibe(description="Test mermaid"),
            root_nodes=[node]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID)

        # Verify Mermaid syntax
        assert "graph TD" in diagram
        assert "test-node" in diagram
        assert "[Test node for mermaid]" in diagram
        assert "classDef" in diagram  # Styling classes
        assert ":::pending" in diagram  # Node styling

    def test_mermaid_diagram_with_execution_results(self):
        """Test Mermaid diagram with execution status styling."""

        node1 = WorkflowNode(
            id="completed-node",
            type=WorkflowNodeType.ANALYZE,
            description="Completed analysis"
        )

        node2 = WorkflowNode(
            id="failed-node",
            type=WorkflowNodeType.VALIDATE,
            description="Failed validation"
        )

        plan = WorkflowPlan(
            id="mermaid-status-test",
            source_vibe=Vibe(description="Mermaid status"),
            root_nodes=[node1, node2]
        )

        # Create execution results
        result = ExecutionResult("mermaid-status-test")
        result.add_node_result(NodeResult("completed-node", ExecutionStatus.COMPLETED))
        result.add_node_result(NodeResult("failed-node", ExecutionStatus.FAILED))

        diagram = WorkflowVisualizer.generate_diagram(plan, result, VisualizationFormat.MERMAID)

        # Verify status styling
        assert ":::completed" in diagram
        assert ":::failed" in diagram
        assert "fill:#d4edda" in diagram  # Completed styling
        assert "fill:#f8d7da" in diagram  # Failed styling

    def test_mermaid_diagram_node_shapes(self):
        """Test Mermaid diagram uses correct shapes for node types."""

        nodes = [
            WorkflowNode(id="analyze-1", type=WorkflowNodeType.ANALYZE, description="Analyze"),
            WorkflowNode(id="generate-1", type=WorkflowNodeType.GENERATE, description="Generate"),
            WorkflowNode(id="transform-1", type=WorkflowNodeType.TRANSFORM, description="Transform"),
            WorkflowNode(id="validate-1", type=WorkflowNodeType.VALIDATE, description="Validate"),
            WorkflowNode(id="composite-1", type=WorkflowNodeType.COMPOSITE, description="Composite")
        ]

        plan = WorkflowPlan(
            id="shapes-test",
            source_vibe=Vibe(description="Test shapes"),
            root_nodes=nodes
        )

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID)

        # Verify node shapes
        assert "([Analyze])" in diagram  # Analyze nodes
        assert "[Generate]" in diagram    # Generate nodes
        assert "{{Transform}}" in diagram # Transform nodes
        assert "((Validate))" in diagram  # Validate nodes
        assert "[[Composite]]" in diagram # Composite nodes

    def test_mermaid_diagram_with_dependencies(self):
        """Test Mermaid diagram shows dependencies as arrows."""

        step1 = WorkflowNode(
            id="step-1",
            type=WorkflowNodeType.ANALYZE,
            description="First"
        )

        step2 = WorkflowNode(
            id="step-2",
            type=WorkflowNodeType.GENERATE,
            description="Second",
            dependencies=["step-1"]
        )

        step3 = WorkflowNode(
            id="step-3",
            type=WorkflowNodeType.VALIDATE,
            description="Third",
            dependencies=["step-2"]
        )

        plan = WorkflowPlan(
            id="mermaid-deps",
            source_vibe=Vibe(description="Dependencies"),
            root_nodes=[step1, step2, step3]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID)

        # Verify dependency arrows
        assert "step-1 --> step-2" in diagram
        assert "step-2 --> step-3" in diagram

    def test_mermaid_long_description_truncation(self):
        """Test Mermaid diagram truncates long node descriptions."""

        long_description = "This is a very long description that should be truncated for display purposes"

        node = WorkflowNode(
            id="long-node",
            type=WorkflowNodeType.GENERATE,
            description=long_description
        )

        plan = WorkflowPlan(
            id="truncation-test",
            source_vibe=Vibe(description="Test truncation"),
            root_nodes=[node]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID)

        # Should be truncated to 30 characters + "..."
        assert "This is a very long descriptio..." in diagram
        assert len("This is a very long descriptio") == 30  # Verify truncation logic

    def test_all_node_collection(self):
        """Test _collect_all_nodes correctly finds all nodes in hierarchy."""

        # Create complex hierarchy
        grandchild = WorkflowNode(
            id="grandchild",
            type=WorkflowNodeType.VALIDATE,
            description="Grandchild node"
        )

        child1 = WorkflowNode(
            id="child-1",
            type=WorkflowNodeType.ANALYZE,
            description="First child",
            children=[grandchild]
        )

        child2 = WorkflowNode(
            id="child-2",
            type=WorkflowNodeType.GENERATE,
            description="Second child"
        )

        root = WorkflowNode(
            id="root",
            type=WorkflowNodeType.COMPOSITE,
            description="Root node",
            children=[child1, child2]
        )

        all_nodes = WorkflowVisualizer._collect_all_nodes([root])

        # Should find all 4 nodes
        assert len(all_nodes) == 4
        node_ids = [node.id for node in all_nodes]
        assert "root" in node_ids
        assert "child-1" in node_ids
        assert "child-2" in node_ids
        assert "grandchild" in node_ids

    def test_unsupported_format_error(self):
        """Test that unsupported formats raise ValueError."""

        node = WorkflowNode(
            id="test",
            type=WorkflowNodeType.GENERATE,
            description="Test"
        )

        plan = WorkflowPlan(
            id="test",
            source_vibe=Vibe(description="Test"),
            root_nodes=[node]
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            WorkflowVisualizer.generate_diagram(plan, format="unsupported")

    def test_empty_workflow_handling(self):
        """Test visualization handles empty workflows gracefully."""

        plan = WorkflowPlan(
            id="empty-test",
            source_vibe=Vibe(description="Empty workflow"),
            root_nodes=[]
        )

        diagram = WorkflowVisualizer.generate_diagram(plan)

        # Should still show header information
        assert "Workflow Plan: empty-test" in diagram
        assert "Source Vibe: Empty workflow" in diagram
        # But no node structure
        assert "├──" not in diagram
        assert "└──" not in diagram