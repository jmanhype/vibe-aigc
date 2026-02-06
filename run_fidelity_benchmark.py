"""Run the fidelity benchmark — Creative Unit Tests for vibe-aigc."""

import asyncio
import json
from vibe_aigc.fidelity import FidelityBenchmark, measure_fidelity
from vibe_aigc.discovery import Capability

async def main():
    print("=" * 60)
    print("VIBE-AIGC FIDELITY BENCHMARK")
    print("Creative Unit Tests — Paper Section 7")
    print("=" * 60)
    print()
    
    benchmark = FidelityBenchmark(
        comfyui_url="http://192.168.1.143:8188",
        max_attempts_per_run=2,
        quality_threshold=7.0
    )
    
    print("Initializing...")
    await benchmark.initialize()
    print()
    
    # Test prompt
    prompt = "cyberpunk samurai warrior standing in neon-lit Tokyo street, rain, cinematic lighting, detailed armor, moody atmosphere"
    
    print("Running fidelity benchmark...")
    print(f"Prompt: {prompt}")
    print(f"Runs: 3")
    print()
    
    report = await benchmark.run(
        prompt=prompt,
        capability=Capability.TEXT_TO_IMAGE,
        num_runs=3,
        width=768,
        height=512,
        steps=20,
        cfg=7.0
    )
    
    print()
    print(report.summary())
    
    # Save report
    with open("fidelity_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print("\nReport saved to fidelity_report.json")

if __name__ == "__main__":
    asyncio.run(main())
