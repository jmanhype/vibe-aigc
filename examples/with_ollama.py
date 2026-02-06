"""Example: Using MetaPlanner with local Ollama LLM.

This demonstrates how to use vibe-aigc without any paid API keys
by leveraging a local Ollama instance.

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull qwen2.5:14b
3. Run Ollama: ollama serve

Usage:
    python -m examples.with_ollama
"""

import asyncio
from vibe_aigc import Vibe, MetaPlanner, LLMConfig


async def main():
    # Configure for local Ollama
    # For remote Ollama (e.g., on your GPU server):
    #   LLMConfig.for_ollama(host="http://192.168.1.143:11434")
    
    config = LLMConfig.for_ollama(
        host="http://localhost:11434",
        model="qwen2.5:14b"  # or qwen2.5:7b for faster inference
    )
    
    # Create MetaPlanner with Ollama backend
    planner = MetaPlanner(llm_config=config)
    
    # Define your creative intent
    vibe = Vibe(
        description="create a music video with synthwave aesthetics",
        style="retro, neon, 80s",
        domain="music_video",
        constraints=["60 seconds", "looping"],
    )
    
    # Generate workflow plan
    print("Generating workflow plan...")
    plan = await planner.plan(vibe)
    
    print(f"\nPlan: {plan.id}")
    print(f"Estimated duration: {plan.estimated_total_duration}s")
    print("\nWorkflow:")
    for i, node in enumerate(plan.root_nodes, 1):
        print(f"{i}. [{node.type.value}] {node.description}")
        for j, child in enumerate(node.children, 1):
            print(f"   {i}.{j}. [{child.type.value}] {child.description}")


if __name__ == "__main__":
    asyncio.run(main())
