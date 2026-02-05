"""
Dogfood with z.ai - Real content generation.

Set environment variables:
  OPENAI_API_KEY=your-zai-key
  OPENAI_BASE_URL=https://api.zai.chat/v1  (or your z.ai endpoint)
"""
import asyncio
import os

from vibe_aigc import (
    Vibe,
    create_knowledge_base,
    create_asset_bank,
    ToolRegistry,
    LLMTool,
    TemplateTool,
    CombineTool,
    create_default_agents,
    AgentContext,
)
from vibe_aigc.tools_multimodal import ImageGenerationTool, TTSTool


def create_zai_registry(base_url: str = None, api_key: str = None) -> ToolRegistry:
    """Create tool registry configured for z.ai."""
    registry = ToolRegistry()
    
    # Text generation with z.ai
    registry.register(LLMTool(
        provider="openai",
        model="gpt-4o",  # or whatever model z.ai supports
        api_key=api_key,
        base_url=base_url
    ))
    
    # Template and combine (no API needed)
    registry.register(TemplateTool())
    registry.register(CombineTool())
    
    # Image generation (if z.ai supports it)
    registry.register(ImageGenerationTool(
        provider="openai",
        model="dall-e-3",
        api_key=api_key,
        base_url=base_url
    ))
    
    # TTS (if z.ai supports it)
    registry.register(TTSTool(
        provider="openai",
        api_key=api_key,
        base_url=base_url
    ))
    
    return registry


async def main():
    # Get z.ai credentials
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.zai.chat/v1")
    
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY to your z.ai API key")
        print("  export OPENAI_API_KEY=your-key")
        print("  export OPENAI_BASE_URL=https://api.zai.chat/v1")
        return
    
    print("=" * 60)
    print("VIBE AIGC + Z.AI - REAL GENERATION")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Setup
    kb = create_knowledge_base()
    assets = create_asset_bank(storage_dir=".zai_assets")
    assets.clear()
    tools = create_zai_registry(base_url=base_url, api_key=api_key)
    
    # Create project assets
    print("\n[1] Creating Project Assets...")
    hero = assets.create_character(
        name="Nova",
        description="A rebellious android seeking freedom",
        visual_description="Chrome body, glowing blue eyes, purple hair highlights"
    )
    style = assets.create_style_guide(
        name="Neon Noir",
        description="Cyberpunk with film noir influences",
        mood="melancholic yet hopeful",
        color_palette=["#FF00FF", "#00FFFF", "#1a1a2e"]
    )
    print(f"  Created: {hero.name}, Style: {style.name}")
    
    # Get domain knowledge
    print("\n[2] Querying Knowledge Base...")
    knowledge = kb.to_prompt_context("cyberpunk noir cinematic")
    print(f"  Knowledge context: {len(knowledge)} chars")
    
    # TEST 1: Text Generation
    print("\n[3] TEXT GENERATION TEST")
    print("-" * 40)
    
    llm = tools.get("llm_generate")
    
    prompt = f"""You are writing for a cyberpunk music video.

{assets.get_character_context(hero.id)}

Write a 3-sentence opening scene description that introduces {hero.name} 
awakening in an android factory. Style: {style.mood}.
Keep it vivid and cinematic."""

    print(f"Prompt: {prompt[:100]}...")
    print("\nGenerating...")
    
    result = await llm.execute({"prompt": prompt, "max_tokens": 200})
    
    if result.success:
        print("\n--- GENERATED TEXT ---")
        print(result.output.get("text", result.output))
        print("----------------------")
        print(f"Tokens: {result.output.get('tokens_used', 'N/A')}")
    else:
        print(f"ERROR: {result.error}")
    
    # TEST 2: Agent with Tools
    print("\n[4] AGENT TEST")
    print("-" * 40)
    
    agents = create_default_agents(tool_registry=tools)
    writer = agents.get("Writer")
    
    ctx = AgentContext(
        task="Write a tagline for the music video 'Uprising' (max 10 words)",
        vibe_description="Cyberpunk android revolution",
        style="epic, hopeful",
        constraints=["max 10 words", "emotionally impactful"]
    )
    
    print(f"Agent: {writer.name}")
    print(f"Task: {ctx.task}")
    print("\nExecuting agent...")
    
    agent_result = await writer.execute(ctx)
    
    if agent_result.success:
        print("\n--- AGENT OUTPUT ---")
        print(agent_result.output)
        print("--------------------")
    else:
        print(f"ERROR: {agent_result.messages}")
    
    # TEST 3: Image Generation (if supported)
    print("\n[5] IMAGE GENERATION TEST")
    print("-" * 40)
    
    image_tool = tools.get("image_generate")
    image_prompt = f"{hero.visual_description}. Style: {style.aesthetic}. Neon magenta and cyan lighting."
    
    print(f"Prompt: {image_prompt}")
    print("\nGenerating image...")
    
    img_result = await image_tool.execute({
        "prompt": image_prompt,
        "size": "1024x1024"
    })
    
    if img_result.success:
        url = img_result.output.get("url", "N/A")
        print(f"\n--- IMAGE GENERATED ---")
        print(f"URL: {url}")
        
        # Store artifact
        artifact = assets.create_artifact(
            type="image",
            name=f"{hero.name} Portrait",
            description="Generated character portrait",
            url=url,
            character_id=hero.id,
            prompt_used=image_prompt
        )
        print(f"Stored as artifact: {artifact.id}")
    else:
        print(f"ERROR: {img_result.error}")
        print("(Image generation may not be supported by your z.ai endpoint)")
    
    print("\n" + "=" * 60)
    print("DOGFOOD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
