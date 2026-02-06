"""
Full E2E Demo - vibe-aigc 0.7.0
Shows: KnowledgeBase -> Pipeline -> Multiple Tools -> Output
"""
import asyncio

async def main():
    print("="*60)
    print("VIBE-AIGC v0.7.0 - FULL E2E DEMO")
    print("="*60)
    print()
    
    # 1. Setup
    from vibe_aigc.knowledge import create_knowledge_base
    from vibe_aigc.tools import create_default_registry
    from vibe_aigc.pipeline import Pipeline, PipelineStep
    
    kb = create_knowledge_base()
    registry = create_default_registry("http://192.168.1.143:8188")
    
    print(f"[SETUP] KnowledgeBase: {len(kb._domains)} domains")
    print(f"[SETUP] ToolRegistry: {len(registry._tools)} tools")
    print()
    
    # 2. Query knowledge for "ghibli" style
    print("[KNOWLEDGE] Querying 'ghibli' aesthetic...")
    ghibli = kb.query("ghibli")
    if ghibli:
        specs = ghibli.get('technical_specs', {})
        tags = specs.get('sd_prompt_tags', [])[:5]
        print(f"  Tags: {tags}")
    print()
    
    # 3. Build enhanced prompt
    base_prompt = "a girl walking through a flower meadow"
    enhanced = base_prompt
    if ghibli:
        specs = ghibli.get('technical_specs', {})
        if specs.get('sd_prompt_tags'):
            enhanced += ", " + ", ".join(specs['sd_prompt_tags'][:6])
        if specs.get('color_palette'):
            enhanced += ", " + ", ".join(specs['color_palette'][:3])
    
    print("[PROMPT] Building enhanced prompt...")
    print(f"  Base: {base_prompt}")
    print(f"  Enhanced: {enhanced}")
    print()
    
    # 4. Generate image
    print("[TOOL: image_generation] Generating Ghibli-style image...")
    image_tool = registry.get("image_generation")
    img_result = await image_tool.execute({
        "prompt": enhanced,
        "negative_prompt": "realistic, photo, 3d render",
        "width": 768,
        "height": 512,
        "steps": 20,
        "cfg": 7.0
    })
    
    if not img_result.success:
        print(f"  FAILED: {img_result.error}")
        return
    
    image_url = img_result.output.get("image_url")
    print(f"  SUCCESS: {image_url}")
    print(f"  Quality: {img_result.output.get('quality_score')}/10")
    print()
    
    # 5. Caption the image
    print("[TOOL: caption] Describing generated image...")
    caption_tool = registry.get("caption")
    if caption_tool:
        cap_result = await caption_tool.execute({"image_url": image_url})
        if cap_result.success:
            print(f"  Caption: {cap_result.output.get('caption', 'N/A')[:100]}...")
        else:
            print(f"  (caption skipped: {cap_result.error})")
    print()
    
    # 6. Generate video from image (I2V mode)
    print("[TOOL: video_generation] Animating image (I2V)...")
    video_tool = registry.get("video_generation")
    vid_result = await video_tool.execute({
        "prompt": "gentle breeze, flowing hair, swaying flowers, peaceful movement",
        "image_url": image_url,  # Pass the generated image to animate
        "width": 832,
        "height": 480,
        "frames": 33
    })
    
    if vid_result.success:
        video_url = vid_result.output.get("video_url")
        print(f"  SUCCESS: {video_url}")
    else:
        print(f"  (video skipped: {vid_result.error})")
        video_url = None
    print()
    
    # 7. Convert to GIF if video succeeded
    if video_url:
        print("[TOOL: video_to_gif] Converting to GIF...")
        gif_tool = registry.get("video_to_gif")
        if gif_tool:
            gif_result = await gif_tool.execute({
                "video_url": video_url,
                "fps": 12,
                "width": 480
            })
            if gif_result.success:
                print(f"  SUCCESS: {gif_result.output.get('gif_url')}")
            else:
                print(f"  (gif skipped: {gif_result.error})")
    print()
    
    # 8. Summary
    print("="*60)
    print("E2E DEMO COMPLETE")
    print("="*60)
    print()
    print("Pipeline executed:")
    print("  1. KnowledgeBase query ('ghibli') -> technical specs")
    print("  2. Prompt enhancement -> added style tags")
    print("  3. image_generation -> Ghibli-style image")
    print("  4. caption -> described the image")
    print("  5. video_generation -> animated the scene")
    print("  6. video_to_gif -> converted for sharing")
    print()
    print("This is vibe-aigc v0.7.0 working end-to-end.")

if __name__ == "__main__":
    asyncio.run(main())
