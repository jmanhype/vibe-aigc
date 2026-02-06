"""Test the new workflow-as-tool registry."""

import asyncio
from pathlib import Path
from vibe_aigc.workflow_registry import (
    create_registry, 
    WorkflowRegistry,
    WorkflowRunner,
    WorkflowCapability
)

async def test():
    print("=" * 60)
    print("WORKFLOW REGISTRY TEST")
    print("=" * 60)
    print()
    
    # Create registry with local workflows + 3090 connection
    registry = create_registry(
        workflow_dirs=['./workflows'],
        comfyui_url="http://192.168.1.143:8188"
    )
    
    # Discover local workflows
    print("1. Discovering local workflows...")
    registry.discover()
    print()
    
    # Show status
    print("2. Registry Status:")
    print(registry.status())
    
    # Try to capture current workflow from 3090
    print("3. Capturing current 3090 workflow via ComfyPilot...")
    current = await registry.capture_current()
    if current:
        print(f"   Name: {current.name}")
        print(f"   Capabilities: {[c.value for c in current.capabilities]}")
        print(f"   Required models: {current.required_models[:3]}...")
    else:
        print("   No workflow loaded in browser")
    print()
    
    # Get workflows for video generation
    print("4. Finding workflows for IMAGE_TO_VIDEO...")
    video_workflows = registry.get_for_capability(WorkflowCapability.IMAGE_TO_VIDEO)
    for wf in video_workflows:
        print(f"   • {wf.name}: {wf.description or 'No description'}")
    print()
    
    # Show how parameterization would work
    if video_workflows:
        print("5. Parameterization example:")
        wf = video_workflows[0]
        print(f"   Workflow: {wf.name}")
        print(f"   Parameters would be injected into prompt/seed/etc nodes")
        print("   (Actual execution skipped for this test)")

if __name__ == '__main__':
    asyncio.run(test())
