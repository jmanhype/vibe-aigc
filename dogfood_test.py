"""Dogfood test - prove the full architecture works."""
import asyncio

async def dogfood():
    print('='*60)
    print('DOGFOOD TEST - Full Paper Architecture')
    print('='*60)
    print()
    
    # 1. Test KnowledgeBase with new aesthetics
    from vibe_aigc.knowledge import create_knowledge_base
    kb = create_knowledge_base()
    
    print('[1] KnowledgeBase - New Aesthetics')
    for style in ['blade runner', 'cyberpunk', 'ghibli', 'noir']:
        result = kb.query(style)
        if result:
            tags = result.get('technical_specs', {}).get('sd_prompt_tags', [])[:3]
            print(f'    {style}: {tags}')
        else:
            print(f'    {style}: NOT FOUND')
    print()
    
    # 2. Test ToolRegistry with all tools
    from vibe_aigc.tools import create_default_registry
    registry = create_default_registry('http://192.168.1.143:8188')
    tools = [t.name for t in registry.list_tools()]
    print(f'[2] ToolRegistry - {len(tools)} tools')
    print(f'    {tools}')
    print()
    
    # 3. Test Pipeline chaining
    from vibe_aigc.pipeline import Pipeline, PipelineStep, PipelineBuilder
    print('[3] Pipeline - Building image->video chain')
    builder = PipelineBuilder('dogfood_test', registry)
    print(f'    Builder created: {builder.name}')
    print()
    
    # 4. Test Ollama connection
    print('[4] Ollama - Checking 3090')
    from vibe_aigc.llm import check_ollama_available, list_ollama_models
    available = await check_ollama_available('http://192.168.1.143:11434')
    print(f'    Available: {available}')
    if available:
        models = await list_ollama_models('http://192.168.1.143:11434')
        print(f'    Models: {len(models)} found')
    print()
    
    # 5. Test VLM refinement
    print('[5] VLM Refinement - Testing prompt fix')
    from vibe_aigc.vlm_feedback import VLMFeedback, FeedbackResult, MediaType
    vlm = VLMFeedback()
    mock_feedback = FeedbackResult(
        quality_score=5.0,
        media_type=MediaType.IMAGE,
        description='Lighting is flat, hands look wrong',
        strengths=['good composition'],
        weaknesses=['lighting is flat', 'hands are distorted']
    )
    refined = vlm.refine_prompt('a warrior standing', mock_feedback)
    print(f'    Original: a warrior standing')
    print(f'    Refined:  {refined}')
    print()
    
    # 6. ACTUAL GENERATION with blade runner style
    print('[6] GENERATION - Blade Runner style image')
    print('    Querying knowledge...')
    br_knowledge = kb.query('blade runner')
    specs = br_knowledge.get('technical_specs', {}) if br_knowledge else {}
    
    base_prompt = 'a detective in a dark alley'
    enhanced = base_prompt
    if specs.get('sd_prompt_tags'):
        enhanced += ', ' + ', '.join(specs['sd_prompt_tags'][:5])
    if specs.get('lighting'):
        enhanced += ', ' + ', '.join(specs['lighting'][:2])
    
    print(f'    Enhanced prompt: {enhanced[:80]}...')
    
    image_tool = registry.get('image_generation')
    if image_tool:
        print('    Generating...')
        result = await image_tool.execute({
            'prompt': enhanced,
            'negative_prompt': 'blurry, distorted',
            'width': 768,
            'height': 512,
            'steps': 20
        })
        print(f'    Success: {result.success}')
        if result.success:
            print(f'    URL: {result.output.get("image_url")}')
            print(f'    Quality: {result.output.get("quality_score")}/10')
        else:
            print(f'    Error: {result.error}')
    print()
    
    print('='*60)
    print('DOGFOOD COMPLETE')
    print('='*60)

if __name__ == '__main__':
    asyncio.run(dogfood())
