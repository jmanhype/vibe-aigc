"""Test the general, constraint-aware architecture."""

import asyncio
from vibe_aigc.discovery import discover_system, Capability
from vibe_aigc.composer_general import create_composer
from vibe_aigc.vibe_backend import VibeBackend, GenerationRequest

async def test():
    print("=" * 60)
    print("GENERAL ARCHITECTURE TEST")
    print("Paper-aligned: Discover -> Match -> Compose -> Execute")
    print("=" * 60)
    print()
    
    # Test on 3090
    comfyui_url = "http://192.168.1.143:8188"
    
    print("1. DISCOVERING SYSTEM CAPABILITIES...")
    print("-" * 40)
    caps = await discover_system(comfyui_url)
    print(caps.summary())
    print()
    
    print("2. HARDWARE CONSTRAINTS...")
    print("-" * 40)
    print(f"   GPU: {caps.hardware.gpu_name}")
    print(f"   VRAM: {caps.hardware.vram_total_gb:.1f}GB")
    print(f"   Can run 8GB model: {caps.hardware.can_run(8.0)}")
    print(f"   Can run 24GB model: {caps.hardware.can_run(24.0)}")
    print()
    
    print("3. AVAILABLE CAPABILITIES...")
    print("-" * 40)
    for cap in Capability:
        if cap != Capability.UNKNOWN:
            has = caps.has_capability(cap)
            models = caps.get_models_for(cap)
            status = f"YES ({len(models)} models)" if has else "NO"
            print(f"   {cap.value}: {status}")
    print()
    
    print("4. CREATING GENERAL COMPOSER...")
    print("-" * 40)
    composer = create_composer(caps)
    
    # Test composing different workflows
    for cap in [Capability.TEXT_TO_IMAGE, Capability.TEXT_TO_VIDEO]:
        if caps.has_capability(cap):
            model = caps.get_models_for(cap)[0] if caps.get_models_for(cap) else None
            if model:
                print(f"   Composing {cap.value} with {model.filename[:40]}...")
                workflow = composer.compose_for_capability(
                    cap, 
                    "test prompt",
                    negative_prompt="bad quality"
                )
                if workflow:
                    print(f"   -> Created workflow with {len(workflow)} nodes")
                else:
                    print(f"   -> Could not compose (missing nodes)")
    print()
    
    print("5. VIBE BACKEND (UNIFIED)...")
    print("-" * 40)
    backend = VibeBackend(
        comfyui_url=comfyui_url,
        enable_vlm=True,
        max_attempts=3,
        quality_threshold=7.0
    )
    await backend.initialize()
    print()
    
    print("6. GENERATION FLOW...")
    print("-" * 40)
    print("""
    User: "cyberpunk samurai"
          |
          v
    [1] DISCOVER: What does user's system have?
        -> GPU: {gpu}
        -> Capabilities: {caps}
          |
          v
    [2] MATCH: Can we do this request?
        -> text_to_video: {has_video}
          |
          v
    [3] COMPOSE: Build workflow from available nodes
        -> Using discovered nodes, not hardcoded
          |
          v
    [4] EXECUTE: Run via ComfyUI
          |
          v
    [5] EVALUATE: VLM scores output
          |
          v
    [6] REFINE: If score < 7, improve prompt and retry
    """.format(
        gpu=caps.hardware.gpu_name,
        caps=len(caps.capabilities),
        has_video="YES" if caps.has_capability(Capability.TEXT_TO_VIDEO) else "NO"
    ))
    
    print("=" * 60)
    print("ARCHITECTURE SUMMARY")
    print("=" * 60)
    print("""
    GENERAL (Paper-Aligned):
    - NO hardcoded model patterns
    - NO hardcoded node types
    - DISCOVERS from ComfyPilot
    - ADAPTS to user's setup
    - COMPOSES from available tools
    
    Works on:
    - 4GB laptop with SD 1.5
    - 8GB desktop with SDXL
    - 24GB workstation with Wan
    - ANY ComfyUI setup
    """)

if __name__ == "__main__":
    asyncio.run(test())
