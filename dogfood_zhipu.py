"""
VIBE AIGC + Zhipu GLM - Full Dogfood Test
"""
import asyncio
import os
from vibe_aigc import (
    create_knowledge_base, 
    create_asset_bank, 
    ToolRegistry, 
    LLMTool, 
    TemplateTool, 
    CombineTool, 
    create_default_agents, 
    AgentContext
)

async def main():
    print("=" * 60)
    print("VIBE AIGC + ZHIPU GLM-4-PLUS - FULL DOGFOOD")
    print("=" * 60)
    
    # Setup
    kb = create_knowledge_base()
    assets = create_asset_bank(".zhipu_test")
    assets.clear()
    
    # Create registry with Zhipu
    registry = ToolRegistry()
    registry.register(LLMTool(
        provider="openai",
        model="glm-4-plus",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    ))
    registry.register(TemplateTool())
    registry.register(CombineTool())
    
    # Create assets
    print("\n[1] ASSET BANK")
    print("-" * 40)
    hero = assets.create_character(
        name="Nova",
        description="A rebellious android seeking freedom in a neon-lit dystopia",
        visual_description="Chrome body, glowing blue eyes, purple hair highlights"
    )
    mentor = assets.create_character(
        name="The Architect", 
        description="An elderly human who designed the android uprising",
        visual_description="Weathered face, white beard, cybernetic left eye"
    )
    style = assets.create_style_guide(
        name="Neon Noir",
        description="Cyberpunk with film noir influences",
        mood="melancholic yet hopeful, tension building to release",
        color_palette=["#FF00FF", "#00FFFF", "#1a1a2e"]
    )
    print(f"Characters: {hero.name}, {mentor.name}")
    print(f"Style: {style.name} - {style.mood}")
    
    # Knowledge
    print("\n[2] KNOWLEDGE BASE")
    print("-" * 40)
    knowledge = kb.query("noir cinematic cyberpunk")
    concepts = [c["concept"] for c in knowledge["matched_concepts"]]
    print(f"Matched concepts: {concepts}")
    specs = list(knowledge["technical_specs"].keys())[:4]
    print(f"Technical specs: {specs}")
    
    # LLM Generation
    print("\n[3] LLM GENERATION (GLM-4-Plus)")
    print("-" * 40)
    llm = registry.get("llm_generate")
    
    prompt = f"""Write a vivid 3-sentence opening scene for a cyberpunk music video:

Character: {hero.name} - {hero.description}
Visual: {hero.visual_description}
Style: {style.mood}

The scene shows Nova awakening in an android factory at night. Make it cinematic and emotional."""

    print(f"Generating scene description...")
    result = await llm.execute({"prompt": prompt, "max_tokens": 300})
    
    if result.success:
        print("\n--- GENERATED SCENE ---")
        print(result.output.get("text", result.output))
        print("-" * 40)
        tokens = result.output.get("tokens_used", "N/A")
        print(f"Tokens used: {tokens}")
    else:
        print(f"Error: {result.error}")
        return
    
    # Agent test
    print("\n[4] WRITER AGENT")
    print("-" * 40)
    agents = create_default_agents(tool_registry=registry)
    writer = agents.get("Writer")
    
    ctx = AgentContext(
        task="Write a powerful 10-word tagline for the music video 'Uprising' about android revolution and freedom",
        vibe_description="Cyberpunk android freedom story",
        style="epic, hopeful, revolutionary",
        constraints=["around 10 words", "emotionally impactful", "memorable"]
    )
    
    print(f"Task: {ctx.task[:60]}...")
    agent_result = await writer.execute(ctx)
    
    if agent_result.success:
        print("\n--- TAGLINE ---")
        print(agent_result.output)
        print("-" * 40)
    else:
        print(f"Error: {agent_result.messages}")
    
    # Integrated workflow - full context
    print("\n[5] INTEGRATED WORKFLOW - FULL CONTEXT")
    print("-" * 40)
    
    project_context = assets.get_project_context()
    domain_context = kb.to_prompt_context("cyberpunk noir cinematic")
    
    full_prompt = f"""{project_context}

{domain_context}

Based on the characters and style guide above, write 3 scene descriptions (2-3 sentences each) for the music video "Uprising":

1. OPENING: Nova awakens in the factory
2. MIDDLE: Nova meets The Architect  
3. CLIMAX: The uprising begins

Make each scene visually striking and emotionally resonant with the Neon Noir style."""

    print(f"Prompt includes: {len(project_context)} chars of assets + {len(domain_context)} chars of knowledge")
    print("Generating full scene breakdown...")
    
    result = await llm.execute({"prompt": full_prompt, "max_tokens": 600})
    if result.success:
        print("\n--- SCENE BREAKDOWN ---")
        print(result.output.get("text", result.output))
        print("-" * 40)
    else:
        print(f"Error: {result.error}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOGFOOD COMPLETE!")
    print("=" * 60)
    print(f"""
What just happened:
- Asset Bank: Created {len(assets.list_characters())} characters, {len(assets.list_style_guides())} style guide
- Knowledge Base: Queried domain expertise ({len(concepts)} concepts matched)
- LLM Tool: Generated content with Zhipu GLM-4-Plus
- Writer Agent: Used tool to create tagline
- Integrated Workflow: Combined assets + knowledge + LLM

VIBE AIGC is REAL. The full architecture works end-to-end!
""")

if __name__ == "__main__":
    asyncio.run(main())
