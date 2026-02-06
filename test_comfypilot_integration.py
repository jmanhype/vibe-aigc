"""Test ComfyPilot integration in the workflow backend."""

import asyncio
from vibe_aigc.workflow_backend import WorkflowBackend, create_workflow_backend
from vibe_aigc.model_registry import ModelCapability

async def test():
    print("=" * 60)
    print("COMFYPILOT INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Create backend connected to 3090
    backend = create_workflow_backend(
        comfyui_url="http://192.168.1.143:8188",
        workflow_dirs=['./workflows']
    )
    
    print("1. Initializing (workflows + models)...")
    await backend.initialize()
    print()
    
    print("2. Model Registry Status:")
    caps = backend.model_registry.get_capabilities()
    for cap in caps:
        print(f"   {cap}")
    print()
    
    print("3. Workflow Registry Status:")
    print(backend.registry.status())
    
    print("4. ComfyPilot Tools Available:")
    print("   - call_comfy_pilot(tool, **kwargs)")
    print("   - download_recommended_model(capability)")
    print("   - auto_upgrade(max_downloads=1)")
    print()
    
    print("5. Testing ensure_models (checks if models exist)...")
    has_video = await backend.ensure_models("text_to_video")
    print(f"   text_to_video models: {'YES' if has_video else 'NO (would download)'}")
    print()
    
    print("6. Best models for each capability:")
    for cap in [ModelCapability.TEXT_TO_IMAGE, ModelCapability.TEXT_TO_VIDEO]:
        best = backend.model_registry.get_best_for(cap)
        if best:
            print(f"   {cap.value}: {best.filename}")
        else:
            print(f"   {cap.value}: None (ComfyPilot would download)")
    print()
    
    print("=" * 60)
    print("COMPLETE SYSTEM FLOW:")
    print("=" * 60)
    print("""
    User Request: "cyberpunk samurai video"
           |
           v
    +------------------+
    | WorkflowBackend  |
    +------------------+
           |
           +---> ensure_models() ---> ComfyPilot download if needed
           |
           +---> WorkflowRegistry ---> SELECT pre-made workflow
           |          OR
           +---> WorkflowComposer ---> COMPOSE from atomic tools
           |
           v
    +------------------+
    | ComfyUI API      | (/prompt)
    +------------------+
           |
           v
    +------------------+
    | VLM Feedback     | (Gemini analyzes output)
    +------------------+
           |
           +---> Score >= 7? Done!
           |
           +---> Score < 7? Refine prompt, retry
    """)

if __name__ == '__main__':
    asyncio.run(test())
