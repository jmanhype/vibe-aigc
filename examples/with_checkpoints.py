"""Example with checkpoint and resume functionality."""

import asyncio
from vibe_aigc import Vibe, MetaPlanner


async def main():
    vibe = Vibe(
        description="Generate comprehensive API documentation",
        style="technical, detailed, well-organized",
        constraints=["include code examples", "cover error handling"]
    )

    # Create planner with automatic checkpointing
    planner = MetaPlanner(checkpoint_interval=2)

    print("=== Starting Workflow with Checkpoints ===\n")

    # Execute with checkpoint support
    result = await planner.execute_with_resume(vibe)

    # List all checkpoints
    checkpoints = planner.list_checkpoints()
    print(f"\n=== Checkpoints Created: {len(checkpoints)} ===")
    for cp in checkpoints:
        print(f"  - {cp['checkpoint_id']}: {cp['completed_nodes']} nodes completed")

    # Show results
    summary = result.get_summary()
    print(f"\n=== Execution Complete ===")
    print(f"Status: {summary['status']}")
    print(f"Completed: {summary['completed']}/{summary['total_nodes']} nodes")

    # Demonstrate resuming (simulated)
    if checkpoints:
        print(f"\n=== Resume Example ===")
        print(f"Could resume from checkpoint: {checkpoints[0]['checkpoint_id']}")
        # Uncomment to actually resume:
        # result = await planner.execute_with_resume(
        #     vibe,
        #     checkpoint_id=checkpoints[0]["checkpoint_id"]
        # )

    # Clean up checkpoints
    print("\n=== Cleaning Up Checkpoints ===")
    for cp in checkpoints:
        planner.delete_checkpoint(cp["checkpoint_id"])
        print(f"  Deleted: {cp['checkpoint_id']}")


if __name__ == "__main__":
    asyncio.run(main())
