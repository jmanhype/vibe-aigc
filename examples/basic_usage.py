"""Basic Vibe AIGC usage example."""

import asyncio
from vibe_aigc import Vibe, MetaPlanner


async def main():
    # Define your creative intent as a Vibe
    vibe = Vibe(
        description="Create a blog post about the future of AI agents",
        style="informative, engaging, accessible",
        constraints=["under 1000 words", "include real-world examples"],
        domain="tech content"
    )

    print(f"Created Vibe: {vibe.description}")
    print(f"Style: {vibe.style}")
    print(f"Constraints: {vibe.constraints}")
    print()

    # Create MetaPlanner (requires OPENAI_API_KEY env var)
    planner = MetaPlanner()

    # Generate workflow plan
    print("Generating workflow plan...")
    plan = await planner.plan(vibe)

    print(f"Plan ID: {plan.id}")
    print(f"Root nodes: {len(plan.root_nodes)}")
    for node in plan.root_nodes:
        print(f"  - [{node.type.value}] {node.description}")

    print()

    # Execute the plan
    print("Executing workflow...")
    result = await planner.execute(vibe)

    # Check results
    summary = result.get_summary()
    print(f"\nExecution Complete!")
    print(f"Status: {summary['status']}")
    print(f"Completed: {summary['completed']}/{summary['total_nodes']} nodes")


if __name__ == "__main__":
    asyncio.run(main())
