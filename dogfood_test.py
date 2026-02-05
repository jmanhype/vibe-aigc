"""
Dogfood Test: Use vibe-aigc to generate content about vibe-aigc.
This tests the full architecture end-to-end.
"""
import asyncio
import os
from vibe_aigc import (
    Vibe, MetaPlanner,
    create_knowledge_base, create_default_registry
)

async def main():
    print("=" * 60)
    print("VIBE AIGC DOGFOOD TEST")
    print("=" * 60)
    
    # 1. Test Knowledge Base
    print("\n[1] KNOWLEDGE BASE TEST")
    print("-" * 40)
    kb = create_knowledge_base()
    
    print(f"Domains: {kb.list_domains()}")
    
    # Test the Hitchcock example from the paper
    result = kb.query("Hitchcockian suspense thriller")
    print(f"\nQuery: 'Hitchcockian suspense thriller'")
    print(f"Matched concepts: {[c['concept'] for c in result['matched_concepts']]}")
    print(f"Technical specs: {list(result['technical_specs'].keys())}")
    
    # Show some specs
    if 'camera' in result['technical_specs']:
        print(f"  Camera: {result['technical_specs']['camera']}")
    if 'lighting' in result['technical_specs']:
        print(f"  Lighting: {result['technical_specs']['lighting']}")
    
    # 2. Test Tool Registry
    print("\n[2] TOOL REGISTRY TEST")
    print("-" * 40)
    registry = create_default_registry()
    
    tools = registry.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")
    
    # Test template tool
    template_tool = registry.get("template_fill")
    result = await template_tool.execute({
        "template_name": "social_post",
        "values": {
            "hook": "[rocket] Just shipped vibe-aigc!",
            "body": "A new paradigm for AI content generation.",
            "call_to_action": "Try it: pip install vibe-aigc",
            "hashtags": "#AI #AIGC #Python"
        }
    })
    print(f"\nTemplate tool test:")
    print(result.output["text"].encode('ascii', 'replace').decode())
    
    # 3. Test Combine Tool
    print("\n[3] COMBINE TOOL TEST")
    print("-" * 40)
    combine_tool = registry.get("combine")
    result = await combine_tool.execute({
        "pieces": ["Part 1: Introduction", "Part 2: Features", "Part 3: Conclusion"],
        "separator": "\n\n---\n\n"
    })
    print(f"Combined {result.metadata['piece_count']} pieces")
    print(result.output["text"][:100] + "...")
    
    # 4. Test MetaPlanner with Knowledge + Tools
    print("\n[4] METAPLANNER INTEGRATION TEST")
    print("-" * 40)
    
    # Check if we have an API key for real execution
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    # Create a vibe
    vibe = Vibe(
        description="Write a short promotional tweet about an AI content generation tool",
        style="engaging, technical but accessible",
        constraints=["under 280 characters", "include emoji"],
        domain="writing"
    )
    
    print(f"Vibe: {vibe.description}")
    print(f"Style: {vibe.style}")
    
    if has_api_key:
        # Create planner with full architecture
        planner = MetaPlanner(
            knowledge_base=kb,
            tool_registry=registry
        )
        
        print("\n[Running with real LLM...]")
        try:
            result = await planner.execute(vibe)
            print(f"Status: {result['status']}")
            print(f"Nodes executed: {result['execution_summary']['total_nodes']}")
            for node_id, node_result in result['node_results'].items():
                print(f"  - {node_id}: {node_result['status']}")
                if node_result.get('result'):
                    output = node_result['result']
                    if isinstance(output, dict) and 'text' in output.get('result', {}):
                        print(f"    Output: {output['result']['text'][:100]}...")
        except Exception as e:
            print(f"Execution error: {e}")
    else:
        print("\n[No API key - skipping MetaPlanner test]")
        print("Set OPENAI_API_KEY to test full execution")
    
    # 5. Knowledge Context Generation
    print("\n[5] KNOWLEDGE CONTEXT FOR LLM")
    print("-" * 40)
    context = kb.to_prompt_context("Create a noir cinematic video with minimalist design")
    print(context[:600])
    
    print("\n" + "=" * 60)
    print("DOGFOOD TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
