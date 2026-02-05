"""
Integration example: Using Vibe AIGC with OpenAI for real content generation.

This example shows how to:
1. Configure a custom LLM endpoint
2. Create a Vibe for content generation
3. Execute with progress tracking
4. Handle the results

Prerequisites:
    export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from vibe_aigc import Vibe, MetaPlanner
from vibe_aigc.llm import LLMConfig


async def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY='sk-...'")
        return

    # Configure LLM (optional - uses env var by default)
    llm_config = LLMConfig(
        api_key=api_key,
        model="gpt-4o-mini",  # or "gpt-4o" for better quality
        temperature=0.7,
        max_tokens=4000
    )

    # Define your creative intent
    vibe = Vibe(
        description="Write a technical blog post about building AI agents with Python",
        style="informative, practical, with code examples",
        constraints=[
            "Target audience: intermediate Python developers",
            "Include working code snippets",
            "Cover key concepts: planning, execution, feedback loops",
            "800-1200 words"
        ],
        domain="technical content",
        metadata={
            "format": "markdown",
            "include_sections": ["introduction", "core concepts", "implementation", "conclusion"]
        }
    )

    print("=" * 60)
    print("Vibe AIGC - Integration Example")
    print("=" * 60)
    print(f"\nDescription: {vibe.description}")
    print(f"Style: {vibe.style}")
    print(f"Constraints: {vibe.constraints}")
    print()

    # Progress callback for real-time updates
    def on_progress(event):
        icons = {
            "started": "🚀",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️"
        }
        icon = icons.get(event.event_type, "📍")
        print(f"  {icon} [{event.node_id}] {event.message}")

    # Create planner with custom config
    planner = MetaPlanner(
        llm_config=llm_config,
        progress_callback=on_progress,
        checkpoint_interval=3  # Save progress every 3 nodes
    )

    print("Generating workflow plan...")
    print("-" * 40)

    # Execute with visualization
    result = await planner.execute_with_visualization(vibe)

    # Display results
    print("\n" + "=" * 60)
    print("Execution Results")
    print("=" * 60)

    summary = result.get_summary()
    print(f"\nStatus: {summary['status']}")
    print(f"Nodes completed: {summary['completed']}/{summary['total_nodes']}")

    # Show outputs from each node
    print("\n" + "-" * 40)
    print("Node Outputs:")
    print("-" * 40)

    for node_id, node_result in result.node_results.items():
        print(f"\n[{node_id}] Status: {node_result.status.value}")
        if node_result.output:
            # Truncate long outputs for display
            output = node_result.output
            if len(output) > 500:
                output = output[:500] + "... (truncated)"
            print(f"Output:\n{output}")

    # Cleanup checkpoints on success
    if summary['status'] == 'completed':
        checkpoints = planner.list_checkpoints()
        for cp in checkpoints:
            planner.delete_checkpoint(cp["checkpoint_id"])
        print(f"\nCleaned up {len(checkpoints)} checkpoint(s)")


if __name__ == "__main__":
    asyncio.run(main())
