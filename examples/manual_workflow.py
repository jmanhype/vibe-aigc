"""Example of manually creating workflows for full control."""

from vibe_aigc import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.visualization import WorkflowVisualizer, VisualizationFormat


def main():
    # Define Vibe
    vibe = Vibe(
        description="Build and deploy a web feature",
        style="systematic, thorough"
    )

    # Create nodes manually
    gather = WorkflowNode(
        id="gather",
        type=WorkflowNodeType.ANALYZE,
        description="Gather requirements from stakeholders"
    )

    design = WorkflowNode(
        id="design",
        type=WorkflowNodeType.GENERATE,
        description="Create technical design document",
        dependencies=["gather"]
    )

    implement = WorkflowNode(
        id="implement",
        type=WorkflowNodeType.GENERATE,
        description="Implement the feature",
        dependencies=["design"]
    )

    # Parallel tasks after implementation
    test = WorkflowNode(
        id="test",
        type=WorkflowNodeType.VALIDATE,
        description="Write and run tests",
        dependencies=["implement"]
    )

    document = WorkflowNode(
        id="document",
        type=WorkflowNodeType.GENERATE,
        description="Write documentation",
        dependencies=["implement"]
    )

    # Final deployment depends on both test and document
    deploy = WorkflowNode(
        id="deploy",
        type=WorkflowNodeType.TRANSFORM,
        description="Deploy to production",
        dependencies=["test", "document"]
    )

    # Create plan
    plan = WorkflowPlan(
        id="feature-plan-001",
        source_vibe=vibe,
        root_nodes=[gather, design, implement, test, document, deploy]
    )

    # Visualize
    print("=== Manual Workflow ===\n")
    print(WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII))

    print("\n=== Mermaid Diagram ===\n")
    print(WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID))

    # Show node details
    print("\n=== Node Details ===")
    for node in plan.root_nodes:
        deps = f" (depends on: {', '.join(node.dependencies)})" if node.dependencies else ""
        print(f"  [{node.type.value}] {node.id}: {node.description}{deps}")


if __name__ == "__main__":
    main()
