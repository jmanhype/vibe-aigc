"""True end-to-end test — generate, download, display."""

import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

from vibe_aigc.vibe_backend import VibeBackend, GenerationRequest
from vibe_aigc.discovery import Capability

OUTPUT_DIR = Path("./generated_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

async def main():
    print("=" * 60)
    print("VIBE-AIGC — TRUE END-TO-END TEST")
    print("=" * 60)
    print()
    
    # Initialize
    backend = VibeBackend(
        comfyui_url="http://192.168.1.143:8188",
        enable_vlm=True,
        max_attempts=2,
        quality_threshold=7.0
    )
    
    print("Initializing...")
    await backend.initialize()
    print()
    
    # Generate
    prompt = "cyberpunk samurai warrior in neon Tokyo rain, cinematic, detailed armor, moody"
    
    print(f"Generating: {prompt}")
    print()
    
    result = await backend.generate(GenerationRequest(
        prompt=prompt,
        capability=Capability.TEXT_TO_IMAGE,
        width=768,
        height=512,
        steps=25,
        cfg=7.5
    ))
    
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")
    print(f"Quality Score: {result.quality_score}/10")
    print()
    
    if result.output_url:
        print(f"Remote URL: {result.output_url}")
        
        # Download the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = OUTPUT_DIR / f"e2e_{timestamp}.png"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(result.output_url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    local_path.write_bytes(content)
                    print(f"Downloaded to: {local_path.absolute()}")
    
    if result.feedback:
        print()
        print("VLM FEEDBACK:")
        print(f"  {result.feedback}")
    
    if result.strengths:
        print()
        print("STRENGTHS:")
        for s in result.strengths:
            print(f"  + {s}")
    
    if result.weaknesses:
        print()
        print("WEAKNESSES:")
        for w in result.weaknesses:
            print(f"  - {w}")
    
    if result.prompt_improvements:
        print()
        print("SUGGESTED IMPROVEMENTS:")
        for p in result.prompt_improvements:
            print(f"  → {p}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    print()
    print("=" * 60)
    
    return result, local_path if result.output_url else None

if __name__ == "__main__":
    result, path = asyncio.run(main())
    if path:
        print(f"\nOutput saved: {path}")
