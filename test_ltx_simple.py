"""Test simpler LTX-2 workflow with SamplerCustomAdvanced."""

import asyncio
import aiohttp
import random

async def test():
    url = 'http://192.168.1.143:8188'
    prompt = 'cyberpunk samurai walking through neon tokyo rain, cinematic'
    seed = random.randint(0, 2**32)
    
    # Simpler workflow using SamplerCustomAdvanced
    workflow = {
        # 1. Load checkpoint (MODEL, CLIP=None, VAE)
        '1': {'class_type': 'CheckpointLoaderSimple',
              'inputs': {'ckpt_name': 'ltx-2-19b-distilled-fp8.safetensors'}},
        
        # 2. Load text encoder - Gemma GGUF for LTX-2
        '2': {'class_type': 'CLIPLoaderGGUF',
              'inputs': {
                  'clip_name': 'gemma-3-12b-it-Q4_K_M.gguf',
                  'type': 'ltxv'
              }},
        
        # 3-4. Encode prompts
        '3': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': prompt, 'clip': ['2', 0]}},
        '4': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': 'blurry, static, low quality', 'clip': ['2', 0]}},
        
        # 5. LTX conditioning
        '5': {'class_type': 'LTXVConditioning',
              'inputs': {'positive': ['3', 0], 'negative': ['4', 0], 'frame_rate': 24.0}},
        
        # 6. Empty latent video
        '6': {'class_type': 'EmptyLTXVLatentVideo',
              'inputs': {'width': 512, 'height': 320, 'length': 25, 'batch_size': 1}},
        
        # 7. Model sampling
        '7': {'class_type': 'ModelSamplingLTXV',
              'inputs': {'model': ['1', 0], 'max_shift': 2.05, 'base_shift': 0.95}},
        
        # 8. Scheduler
        '8': {'class_type': 'LTXVScheduler',
              'inputs': {'steps': 20, 'max_shift': 2.05, 'base_shift': 0.95, 'stretch': True, 'terminal': 0.1}},
        
        # 9-11. Sampler components
        '9': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}},
        '10': {'class_type': 'CFGGuider',
               'inputs': {'model': ['7', 0], 'positive': ['5', 0], 'negative': ['5', 1], 'cfg': 3.0}},
        '11': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
        
        # 12. SamplerCustomAdvanced (the proven approach)
        '12': {'class_type': 'SamplerCustomAdvanced',
               'inputs': {
                   'noise': ['9', 0],
                   'guider': ['10', 0],
                   'sampler': ['11', 0],
                   'sigmas': ['8', 0],
                   'latent_image': ['6', 0]
               }},
        
        # 13. Decode
        '13': {'class_type': 'VAEDecode',
               'inputs': {'samples': ['12', 0], 'vae': ['1', 2]}},
        
        # 14. Save
        '14': {'class_type': 'SaveAnimatedWEBP',
               'inputs': {'images': ['13', 0], 'filename_prefix': 'ltx2_test', 'fps': 24.0, 
                          'lossless': False, 'quality': 85, 'method': 'default'}}
    }
    
    print(f'Prompt: {prompt}')
    print(f'Seed: {seed}')
    print(f'Workflow: {len(workflow)} nodes')
    
    async with aiohttp.ClientSession() as session:
        print('\nSubmitting...')
        async with session.post(f'{url}/prompt', json={'prompt': workflow}) as resp:
            result = await resp.json()
            
            if 'error' in result:
                print(f'ERROR: {result["error"]["message"]}')
                if 'node_errors' in result:
                    for nid, err in result['node_errors'].items():
                        print(f'  Node {nid}: {err["errors"][0]["message"]}')
                return
            
            prompt_id = result['prompt_id']
            print(f'Queued: {prompt_id}')
        
        print('Generating...')
        for i in range(180):
            await asyncio.sleep(2)
            async with session.get(f'{url}/history/{prompt_id}') as resp:
                hist = await resp.json()
                if prompt_id in hist:
                    status = hist[prompt_id].get('status', {})
                    if status.get('completed'):
                        outputs = hist[prompt_id].get('outputs', {})
                        for nid, out in outputs.items():
                            if 'images' in out:
                                fn = out['images'][0]['filename']
                                print(f'\nSUCCESS: {url}/view?filename={fn}')
                                return
                    if status.get('status_str') == 'error':
                        msgs = status.get('messages', [])
                        for m in msgs:
                            if m[0] == 'execution_error':
                                print(f'\nERROR: {m[1].get("exception_message", "Unknown")[:500]}')
                        return
            if i % 15 == 0 and i > 0:
                print(f'  ... {i*2}s')

if __name__ == '__main__':
    asyncio.run(test())
