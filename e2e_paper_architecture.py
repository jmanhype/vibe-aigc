"""
End-to-End Test: Paper Architecture

This test demonstrates the paper's architecture:
1. Vibe (high-level creative intent)
2. KnowledgeBase (domain expertise)
3. ToolRegistry (atomic tools including ComfyUI)
4. Direct tool execution with knowledge-enhanced prompts
"""

import asyncio
from vibe_aigc.models import Vibe
from vibe_aigc.knowledge import create_knowledge_base
from vibe_aigc.tools import create_default_registry

async def main():
    print("=" * 60)
    print("VIBE AIGC - PAPER ARCHITECTURE E2E TEST")
    print("=" * 60)
    print()
    
    # Step 1: Express creative intent as a VIBE
    vibe = Vibe(
        description="A cyberpunk samurai emerges from neon rain",
        style="cinematic, Blade Runner aesthetic, dramatic lighting",
        constraints=["moody atmosphere", "high contrast"],
        domain="visual"
    )
    
    print("VIBE (Creative Intent):")
    print(f"  Description: {vibe.description}")
    print(f"  Style: {vibe.style}")
    print(f"  Constraints: {vibe.constraints}")
    print()
    
    # Step 2: Query KnowledgeBase for domain expertise
    kb = create_knowledge_base()
    print(f"KnowledgeBase: {len(kb._domains)} domains loaded")
    print(f"  Domains: {list(kb._domains.keys())}")
    print()
    
    # Query for "cinematic" - this is the paper's key innovation
    print("Querying 'cinematic' for technical specs...")
    cinematic = kb.query("cinematic")
    if cinematic:
        print(f"  Description: {cinematic.get('description', 'N/A')}")
        specs = cinematic.get('technical_specs', {})
        for key, value in specs.items():
            print(f"  {key}: {value}")
    print()
    
    # Query for "Blade Runner" aesthetic
    print("Querying 'blade runner' aesthetic...")
    blade_runner = kb.query("blade runner")
    if blade_runner:
        specs = blade_runner.get('technical_specs', {})
        for key, value in specs.items():
            print(f"  {key}: {value}")
    print()
    
    # Step 3: Create ToolRegistry with ComfyUI tools
    registry = create_default_registry(comfyui_url="http://192.168.1.143:8188")
    print(f"ToolRegistry: {len(registry._tools)} tools registered")
    for spec in registry.list_tools():
        print(f"  - [{spec.category.value}] {spec.name}: {spec.description[:50]}...")
    print()
    
    # Step 4: Build enhanced prompt using knowledge
    base_prompt = vibe.description
    if cinematic:
        specs = cinematic.get('technical_specs', {})
        if 'depth_of_field' in specs:
            base_prompt += f", {specs['depth_of_field']}"
        if 'color_grading' in specs:
            grading = specs['color_grading']
            if isinstance(grading, list):
                base_prompt += f", {', '.join(grading[:2])}"
    
    enhanced_prompt = f"{base_prompt}, {vibe.style}"
    print("Knowledge-Enhanced Prompt:")
    print(f"  Original: {vibe.description}")
    print(f"  Enhanced: {enhanced_prompt}")
    print()
    
    # Step 5: Execute using registered tool
    image_tool = registry.get("image_generation")
    if image_tool:
        print("Executing image_generation tool...")
        print("(This uses VibeBackend -> ComfyUI -> FLUX)")
        print()
        
        result = await image_tool.execute(
            inputs={
                "prompt": enhanced_prompt,
                "negative_prompt": "blurry, distorted, ugly",
                "width": 768,
                "height": 512,
                "steps": 20,
                "cfg": 7.0
            },
            context={
                "technical_specs": cinematic.get('technical_specs', {}) if cinematic else {}
            }
        )
        
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Success: {result.success}")
        if result.success:
            print(f"Image URL: {result.output.get('image_url')}")
            print(f"Quality Score: {result.output.get('quality_score')}")
            if result.output.get('feedback'):
                print(f"VLM Feedback: {result.output.get('feedback')[:100]}...")
        else:
            print(f"Error: {result.error}")
    else:
        print("image_generation tool not found!")
    
    print()
    print("=" * 60)
    print("This is the PAPER architecture:")
    print("  Vibe -> KnowledgeBase -> Enhanced Prompt -> Tool -> Output")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
