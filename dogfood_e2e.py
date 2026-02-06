"""End-to-end dogfood test of vibe-aigc 0.6.0"""

import asyncio
from vibe_aigc.vibe_backend import VibeBackend, GenerationRequest
from vibe_aigc.discovery import Capability

async def main():
    print("=" * 60)
    print("VIBE-AIGC 0.6.0 — END-TO-END DOGFOOD TEST")
    print("=" * 60)
    print()
    
    # Connect to 3090
    backend = VibeBackend(
        comfyui_url="http://192.168.1.143:8188",
        workflow_dirs=['./workflows'],
        enable_vlm=True,
        max_attempts=2,
        quality_threshold=6.0
    )
    
    print("1. Initializing backend...")
    caps = await backend.initialize()
    print()
    
    print("2. Creating generation request...")
    request = GenerationRequest(
        prompt="cyberpunk samurai warrior walking through neon-lit Tokyo streets at night, rain, cinematic lighting, detailed armor, 4k",
        capability=Capability.TEXT_TO_IMAGE,
        negative_prompt="blurry, low quality, distorted, ugly",
        width=768,
        height=512,
        steps=20,
        cfg=7.0
    )
    print(f"   Prompt: {request.prompt[:50]}...")
    print(f"   Capability: {request.capability.value}")
    print()
    
    print("3. Generating...")
    result = await backend.generate(request)
    print()
    
    print("4. Result:")
    print(f"   Success: {result.success}")
    print(f"   Attempts: {result.attempts}")
    if result.output_url:
        print(f"   Output: {result.output_url}")
    if result.quality_score:
        print(f"   Quality Score: {result.quality_score}/10")
    if result.feedback:
        print(f"   VLM Feedback: {result.feedback[:100]}...")
    if result.error:
        print(f"   Error: {result.error}")
    print()
    
    print("=" * 60)
    if result.success:
        print("DOGFOOD TEST PASSED!")
    else:
        print("DOGFOOD TEST FAILED")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    asyncio.run(main())
