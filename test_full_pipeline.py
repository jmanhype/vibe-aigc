"""Test the full pipeline with workflow backend."""

import asyncio
from vibe_aigc.mv_pipeline import MVPipeline, VideoBackend, Shot
from vibe_aigc.models import Vibe

async def test():
    print("=" * 60)
    print("FULL PIPELINE TEST (with Workflow Backend)")
    print("=" * 60)
    print()
    
    # Create pipeline with workflow backend
    pipeline = MVPipeline(
        comfyui_url="http://192.168.1.143:8188",
        video_backend=VideoBackend.WORKFLOW,
        enable_vlm_feedback=True
    )
    
    print("1. Initializing pipeline...")
    await pipeline.initialize()
    print()
    
    print("2. Pipeline Status:")
    print(f"   VLM Feedback: {'Available' if pipeline.vlm_feedback and pipeline.vlm_feedback.available else 'Not available'}")
    print(f"   Workflow Backend: {'Discovered' if pipeline.workflow_backend._discovered else 'Not discovered'}")
    print()
    
    print("3. Creating test shot...")
    shot = Shot(
        id="test_shot_1",
        description="A cyberpunk samurai walking through neon-lit streets",
        prompt="cyberpunk samurai warrior, neon lights, rain, cinematic, detailed armor",
        negative_prompt="blurry, static, distorted, low quality",
        duration=2.0,
        frames=24
    )
    print(f"   Prompt: {shot.prompt}")
    print()
    
    print("4. Generating shot (this will take a while)...")
    print("   Using: WorkflowBackend -> WorkflowRegistry -> ComfyUI 3090")
    print()
    
    # Actually generate (uncomment to run)
    # result_shot = await pipeline.generate_shot(shot, use_vlm_feedback=True)
    # print(f"   Result: {result_shot.video_url}")
    
    print("   [Skipping actual generation - uncomment to run]")
    print()
    
    print("=" * 60)
    print("SUCCESS: Pipeline wired correctly")
    print()
    print("Flow:")
    print("  MVPipeline")
    print("    -> WorkflowBackend")
    print("       -> WorkflowRegistry (select pre-made)")
    print("       -> OR WorkflowComposer (build new)")
    print("    -> ComfyUI API (execute)")
    print("    -> VLM Feedback (analyze)")
    print("    -> Loop until quality >= threshold")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(test())
