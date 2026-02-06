"""Test the new strategy-based workflow executor."""

import asyncio
from vibe_aigc.workflow_executor import (
    WorkflowExecutor,
    generate_video,
    Capability,
)

async def test_discovery():
    """Test discovery on 3090."""
    print("=" * 60)
    print("TESTING DISCOVERY-BASED ARCHITECTURE")
    print("=" * 60)
    print()
    
    executor = WorkflowExecutor("http://192.168.1.143:8188")
    
    # Step 1: Discover
    print("Step 1: Discovery")
    print("-" * 40)
    await executor.discover()
    print()
    
    # Step 2: Show status
    print("Step 2: Status")
    print("-" * 40)
    print(executor.status())
    print()
    
    # Step 3: List all viable strategies
    print("Step 3: Viable Strategies")
    print("-" * 40)
    from vibe_aigc.workflow_strategies import StrategyFactory
    
    viable = StrategyFactory.get_all_viable(
        executor.discovery.models,
        max_vram=executor.discovery.hardware.vram_free_gb
    )
    
    for strategy in viable:
        models = strategy.select_models(
            executor.discovery.models,
            executor.discovery.hardware.vram_free_gb
        )
        print(f"  {strategy.name}:")
        for role, model in models.items():
            print(f"    {role}: {model.filename}")
    
    if not viable:
        print("  No viable strategies found!")
        print("  Available models:")
        for cat, models in executor.discovery.models.items():
            if models:
                print(f"    {cat}: {[m.filename for m in models[:3]]}")
    print()
    
    # Step 4: Select best strategy
    print("Step 4: Strategy Selection")
    print("-" * 40)
    strategy = executor.select_strategy(
        capability=Capability.TEXT_TO_VIDEO,
        preference="quality"
    )
    
    if strategy:
        print(f"Selected: {strategy.name}")
        
        # Show the workflow that would be built
        from vibe_aigc.workflow_strategies import VideoRequest
        request = VideoRequest(
            prompt="cyberpunk samurai in neon rain",
            frames=25
        )
        
        try:
            workflow = strategy.build_workflow(request, executor.selected_models)
            print(f"Workflow has {len(workflow)} nodes:")
            for nid, node in workflow.items():
                print(f"  {nid}: {node['class_type']}")
        except Exception as e:
            print(f"Workflow build error: {e}")
    else:
        print("No strategy selected")
    print()
    
    return executor


async def test_generation():
    """Test actual video generation."""
    print("=" * 60)
    print("TESTING VIDEO GENERATION")
    print("=" * 60)
    print()
    
    result = await generate_video(
        comfyui_url="http://192.168.1.143:8188",
        prompt="cyberpunk samurai walking through neon tokyo rain, cinematic",
        frames=25,
        preference="quality",
    )
    
    print()
    print("Result:")
    print(f"  Success: {result.success}")
    print(f"  Outputs: {result.outputs}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"  Time: {result.execution_time:.1f}s")
    
    return result


if __name__ == "__main__":
    # First test discovery
    executor = asyncio.run(test_discovery())
    
    # Ask before running actual generation
    print()
    response = input("Run actual generation? (y/n): ").strip().lower()
    if response == "y":
        asyncio.run(test_generation())
