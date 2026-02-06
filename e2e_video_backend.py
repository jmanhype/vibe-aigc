"""True E2E test using VibeBackend for video generation."""

import asyncio
from vibe_aigc.vibe_backend import VibeBackend, GenerationRequest
from vibe_aigc.discovery import Capability

async def main():
    print("=" * 60)
    print("VIBE-AIGC E2E VIDEO TEST (via VibeBackend)")
    print("=" * 60)
    print()
    
    backend = VibeBackend(
        comfyui_url="http://192.168.1.143:8188",
        enable_vlm=False,  # Skip VLM for video
        max_attempts=1
    )
    
    await backend.initialize()
    
    prompt = "cyberpunk samurai warrior in neon rain, cinematic, dramatic"
    
    print(f"\nGenerating video: {prompt}\n")
    
    result = await backend.generate(GenerationRequest(
        prompt=prompt,
        capability=Capability.TEXT_TO_VIDEO,
        width=832,
        height=480,
        frames=33,  # ~2 sec @ 16fps
        steps=20,
        cfg=5.0
    ))
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Output URL: {result.output_url}")
    print(f"Output Path: {result.output_path}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
