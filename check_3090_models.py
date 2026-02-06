import asyncio
from vibe_aigc.model_registry import ModelRegistry, ModelCapability

async def check():
    reg = ModelRegistry(comfyui_url='http://192.168.1.143:8188')
    await reg.refresh()
    
    print('Models on 3090:')
    print('=' * 50)
    
    # Check by category
    for cat in ['checkpoints', 'unet', 'diffusion_models', 'vae', 'clip', 'loras']:
        models = reg.models.get(cat, [])
        if models:
            print(f'\n{cat}: {len(models)} models')
            for m in models[:5]:
                print(f'  - {m.filename}')
            if len(models) > 5:
                print(f'  ... and {len(models)-5} more')
    
    print('\n' + '=' * 50)
    print('Best for each capability:')
    for cap in ModelCapability:
        best = reg.get_best_for(cap)
        if best:
            print(f'  {cap.value}: {best.filename}')

asyncio.run(check())
