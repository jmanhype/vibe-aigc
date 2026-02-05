"""
Full Dogfood Test: Multi-modal vibe-aigc in action.

This demonstrates the complete architecture:
1. Asset Bank - Character and style consistency
2. Knowledge Base - Domain expertise
3. Agents - Specialized roles
4. Tools - Multi-modal generation
"""
import asyncio
import os

from vibe_aigc import (
    # Core
    Vibe,
    # Knowledge
    create_knowledge_base,
    # Tools
    create_full_registry,
    ToolRegistry,
    # Agents
    create_default_agents,
    AgentContext,
    AgentRole,
    WriterAgent,
    ResearcherAgent,
    EditorAgent,
    DirectorAgent,
    DesignerAgent,
    ScreenwriterAgent,
    # Assets
    create_asset_bank,
)


async def main():
    print("=" * 70)
    print("VIBE AIGC v0.2.0 - FULL DOGFOOD TEST")
    print("=" * 70)
    
    # ========== 1. ASSET BANK ==========
    print("\n" + "=" * 70)
    print("[1] ASSET BANK - Character & Style Consistency")
    print("=" * 70)
    
    assets = create_asset_bank(storage_dir=".dogfood_assets")
    assets.clear()  # Fresh start
    
    # Create characters for a music video project
    hero = assets.create_character(
        name="Nova",
        description="A rebellious android seeking freedom in a neon-lit dystopia",
        visual_description="Sleek chrome body, glowing blue eyes, asymmetric haircut with purple highlights",
        personality="Defiant, curious, secretly hopeful"
    )
    print(f"Created character: {hero.name} (ID: {hero.id})")
    print(f"  Visual: {hero.visual_description}")
    
    mentor = assets.create_character(
        name="The Architect",
        description="An elderly human who designed the android uprising",
        visual_description="Weathered face, white beard, cybernetic left eye, worn lab coat",
        personality="Wise, guilt-ridden, protective"
    )
    print(f"Created character: {mentor.name} (ID: {mentor.id})")
    
    # Create style guide
    style = assets.create_style_guide(
        name="Neon Noir",
        description="Cyberpunk aesthetic with film noir influences",
        mood="melancholic yet hopeful, tension building to release",
        aesthetic="high contrast, neon accents against dark backgrounds",
        color_palette=["#FF00FF", "#00FFFF", "#1a1a2e", "#0f0f1a"]
    )
    print(f"\nCreated style guide: {style.name}")
    print(f"  Mood: {style.mood}")
    print(f"  Colors: {style.color_palette}")
    
    # Show project context
    print("\n--- Project Context for Prompts ---")
    context = assets.get_project_context()
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # ========== 2. KNOWLEDGE BASE ==========
    print("\n" + "=" * 70)
    print("[2] KNOWLEDGE BASE - Domain Expertise")
    print("=" * 70)
    
    kb = create_knowledge_base()
    
    # Query for our project style
    query = "noir cinematic cyberpunk music video"
    result = kb.query(query)
    
    print(f"Query: '{query}'")
    print(f"Matched concepts: {[c['concept'] for c in result['matched_concepts']]}")
    print(f"Technical specs extracted:")
    for key, value in list(result['technical_specs'].items())[:5]:
        print(f"  - {key}: {value}")
    
    # Generate LLM context
    llm_context = kb.to_prompt_context(query)
    print(f"\nKnowledge context length: {len(llm_context)} chars")
    
    # ========== 3. TOOL REGISTRY ==========
    print("\n" + "=" * 70)
    print("[3] TOOL REGISTRY - Multi-Modal Capabilities")
    print("=" * 70)
    
    tools = create_full_registry()
    tool_list = tools.list_tools()
    
    print("Available tools:")
    for tool in tool_list:
        print(f"  - {tool.name} ({tool.category.value}): {tool.description[:50]}...")
    
    # Test template tool (no API needed)
    print("\n--- Template Tool Test ---")
    template_tool = tools.get("template_fill")
    result = await template_tool.execute({
        "template_name": "social_post",
        "values": {
            "hook": "Introducing Nova - an android seeking freedom",
            "body": "In a neon-lit dystopia, one android dares to dream. Coming soon: UPRISING - a Vibe AIGC music video.",
            "call_to_action": "Follow for updates",
            "hashtags": "#AI #MusicVideo #Cyberpunk #VibeAIGC"
        }
    })
    if result.success:
        print("Generated social post:")
        print("-" * 40)
        print(result.output["text"])
        print("-" * 40)
    
    # Test combine tool
    print("\n--- Combine Tool Test ---")
    combine_tool = tools.get("combine")
    result = await combine_tool.execute({
        "pieces": [
            "ACT 1: Nova awakens in the factory",
            "ACT 2: She meets The Architect",
            "ACT 3: The uprising begins"
        ],
        "separator": "\n\n"
    })
    print(f"Combined {result.metadata['piece_count']} acts into script outline")
    
    # ========== 4. AGENTS ==========
    print("\n" + "=" * 70)
    print("[4] SPECIALIZED AGENTS - Role-Based Execution")
    print("=" * 70)
    
    agents = create_default_agents(tool_registry=tools)
    
    print("Available agents:")
    for name in agents.list_agents():
        agent = agents.get(name)
        print(f"  - {name} ({agent.role.value})")
    
    # Create a team for our music video
    print("\n--- Creating Production Team ---")
    team = agents.create_team([
        AgentRole.SCREENWRITER,
        AgentRole.DIRECTOR,
        AgentRole.DESIGNER,
        AgentRole.COMPOSER
    ])
    print(f"Team assembled: {[role.value for role in team.keys()]}")
    
    # Test Director coordination
    print("\n--- Director Agent Test ---")
    director = agents.get("Director")
    
    # Add team members to director
    for role, agent in team.items():
        if agent != director:
            director.add_agent(agent)
    
    # Create context with assets
    ctx = AgentContext(
        task="Plan the production of a 60-second music video for 'Uprising'",
        vibe_description="Cyberpunk music video about android revolution",
        style="Neon Noir - high contrast, neon accents, film noir lighting",
        constraints=["60 seconds", "3 acts", "2 main characters"],
        shared_assets={
            "characters": [hero.name, mentor.name],
            "style_guide": style.name
        }
    )
    
    print(f"Task: {ctx.task}")
    print(f"Team: {list(director._managed_agents.keys())}")
    
    # ========== 5. INTEGRATED WORKFLOW ==========
    print("\n" + "=" * 70)
    print("[5] INTEGRATED WORKFLOW - Everything Together")
    print("=" * 70)
    
    # Build a complete prompt using all components
    full_prompt = f"""
{assets.get_project_context()}

{kb.to_prompt_context("cyberpunk noir cinematic")}

TASK: Write the opening scene (15 seconds) of the music video.

Requirements:
- Feature character: {hero.name}
- Setting: Android factory at night
- Mood: {style.mood}
- Visual style: {style.aesthetic}
"""
    
    print("Integrated prompt built from:")
    print("  - Asset Bank: 2 characters, 1 style guide")
    print("  - Knowledge Base: Film + design concepts")
    print(f"  - Total prompt length: {len(full_prompt)} chars")
    
    print("\n--- Sample of Integrated Prompt ---")
    print(full_prompt[:800] + "...")
    
    # ========== 6. API TEST (if keys available) ==========
    print("\n" + "=" * 70)
    print("[6] LIVE API TEST")
    print("=" * 70)
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_replicate = bool(os.getenv("REPLICATE_API_TOKEN"))
    has_brave = bool(os.getenv("BRAVE_API_KEY"))
    
    print(f"API Keys detected:")
    print(f"  - OpenAI: {'Yes' if has_openai else 'No (set OPENAI_API_KEY)'}")
    print(f"  - Replicate: {'Yes' if has_replicate else 'No (set REPLICATE_API_TOKEN)'}")
    print(f"  - Brave Search: {'Yes' if has_brave else 'No (set BRAVE_API_KEY)'}")
    
    if has_openai:
        print("\n--- Live LLM Generation ---")
        llm_tool = tools.get("llm_generate")
        result = await llm_tool.execute({
            "prompt": f"Write a 3-line description for a character named {hero.name}: {hero.description}. Style: {style.aesthetic}",
            "max_tokens": 150
        })
        if result.success:
            print("Generated character description:")
            print(result.output.get("text", result.output))
        else:
            print(f"Error: {result.error}")
    
    if has_openai:
        print("\n--- Live Image Generation Prompt ---")
        # Just show what we WOULD generate
        image_prompt = f"{hero.visual_description}. Style: {style.aesthetic}. Colors: neon magenta and cyan against dark background."
        print(f"Would generate: {image_prompt}")
        
        # Actually try it
        image_tool = tools.get("image_generate")
        print("Attempting DALL-E generation...")
        result = await image_tool.execute({
            "prompt": image_prompt,
            "size": "1024x1024",
            "style": "vivid"
        })
        if result.success:
            print(f"Generated image URL: {result.output.get('url', 'N/A')[:80]}...")
            
            # Store as artifact
            artifact = assets.create_artifact(
                type="image",
                name=f"{hero.name} Portrait",
                description="Character portrait generated by DALL-E",
                url=result.output.get("url"),
                character_id=hero.id,
                prompt_used=image_prompt
            )
            print(f"Stored as artifact: {artifact.id}")
        else:
            print(f"Error: {result.error}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("DOGFOOD TEST COMPLETE")
    print("=" * 70)
    
    print(f"""
Summary:
- Asset Bank: {len(assets.list_characters())} characters, {len(assets.list_style_guides())} style guides
- Knowledge Base: {len(kb.list_domains())} domains
- Tools: {len(tools.list_tools())} tools available
- Agents: {len(agents.list_agents())} agents ready

The full Vibe AIGC architecture is operational!
""")

if __name__ == "__main__":
    asyncio.run(main())
