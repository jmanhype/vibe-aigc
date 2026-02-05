"""Example with workflow visualization."""

import asyncio
from vibe_aigc import Vibe, MetaPlanner
from vibe_aigc.visualization import WorkflowVisualizer, VisualizationFormat


async def main():
    vibe = Vibe(
        description="Design a mobile app onboarding flow",
        style="user-friendly, modern, minimal",
        constraints=["3-5 screens max", "under 30 seconds"]
    )

    # Progress callback for real-time updates
    def on_progress(event):
        emoji = {
            "started": "🚀",
            "completed": "✅", 
            "failed": "❌",
            "skipped": "⏭️"
        }.get(event.event_type, "📍")
        print(f"{emoji} [{event.node_id}] {event.message}")

    planner = MetaPlanner(progress_callback=on_progress)

    # Generate plan first
    print("=== Generating Plan ===\n")
    plan = await planner.plan(vibe)

    # Show ASCII visualization
    print("=== Workflow Diagram (ASCII) ===\n")
    ascii_viz = WorkflowVisualizer.generate_diagram(
        plan,
        format=VisualizationFormat.ASCII
    )
    print(ascii_viz)

    # Show Mermaid visualization
    print("\n=== Workflow Diagram (Mermaid) ===\n")
    mermaid_viz = WorkflowVisualizer.generate_diagram(
        plan,
        format=VisualizationFormat.MERMAID
    )
    print(mermaid_viz)

    # Execute with progress tracking
    print("\n=== Executing Workflow ===\n")
    result = await planner.execute(vibe)

    # Show final visualization with status
    print("\n=== Final Status ===\n")
    final_viz = WorkflowVisualizer.generate_diagram(
        plan,
        execution_result=result,
        format=VisualizationFormat.ASCII
    )
    print(final_viz)


if __name__ == "__main__":
    asyncio.run(main())
