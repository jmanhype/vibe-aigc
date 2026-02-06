import asyncio
import aiohttp
import random

async def generate_ltx_video():
    url = 'http://192.168.1.143:8188'
    prompt = 'cyberpunk samurai walking through neon tokyo rain, cinematic, blade runner'
    seed = random.randint(0, 2**32)
    
    # LTX-2 workflow with separate loaders
    workflow = {
        # Load model (UNET)
        '1': {'class_type': 'UNETLoader', 
              'inputs': {
                  'unet_name': 'ltx-2-19b-distilled-fp8.safetensors',
                  'weight_dtype': 'fp8_e4m3fn'
              }},
        # Load text encoder (T5)
        '2': {'class_type': 'CLIPLoaderGGUF',
              'inputs': {
                  'clip_name': 't5xxl_fp8_e4m3fn_scaled.safetensors',
                  'type': 'ltxv'
              }},
        # Load VAE
        '3': {'class_type': 'VAELoader',
              'inputs': {'vae_name': 'LTX2_video_vae_bf16.safetensors'}},
        
        # Encode prompts
        '4': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': prompt, 'clip': ['2', 0]}},
        '5': {'class_type': 'CLIPTextEncode', 
              'inputs': {'text': 'blurry, static, low quality, watermark', 'clip': ['2', 0]}},
        
        # LTX conditioning
        '6': {'class_type': 'LTXVConditioning',
              'inputs': {'positive': ['4', 0], 'negative': ['5', 0], 'frame_rate': 24}},
        
        # Empty latent video
        '7': {'class_type': 'EmptyLTXVLatentVideo',
              'inputs': {'width': 512, 'height': 320, 'length': 25, 'batch_size': 1}},
        
        # Model sampling
        '8': {'class_type': 'ModelSamplingLTXV',
              'inputs': {'model': ['1', 0], 'max_shift': 2.05, 'base_shift': 0.95}},
        
        # Scheduler
        '9': {'class_type': 'LTXVScheduler',
              'inputs': {'steps': 20, 'max_shift': 2.05, 'base_shift': 0.95, 'stretch': True, 'terminal': 0.1}},
        
        # Sampling
        '10': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}},
        '11': {'class_type': 'BasicGuider', 'inputs': {'model': ['8', 0], 'conditioning': ['6', 0]}},
        '12': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
        '13': {'class_type': 'SamplerCustomAdvanced',
               'inputs': {
                   'noise': ['10', 0],
                   'guider': ['11', 0],
                   'sampler': ['12', 0],
                   'sigmas': ['9', 0],
                   'latent_image': ['7', 0]
               }},
        
        # Decode and save
        '14': {'class_type': 'VAEDecode',
               'inputs': {'samples': ['13', 0], 'vae': ['3', 0]}},
        '15': {'class_type': 'SaveAnimatedWEBP',
               'inputs': {'images': ['14', 0], 'filename_prefix': 'ltx_3090', 'fps': 24.0, 'lossless': False, 'quality': 85, 'method': 'default'}}
    }
    
    async with aiohttp.ClientSession() as session:
        print('>>> Queuing LTX-2 19B on 3090...')
        print(f'    Prompt: {prompt}')
        async with session.post(f'{url}/prompt', json={'prompt': workflow}) as resp:
            result = await resp.json()
            
            if 'error' in result:
                print(f"ERROR: {result['error']['message']}")
                if 'node_errors' in result:
                    for nid, err in result['node_errors'].items():
                        print(f"   Node {nid}: {err['errors'][0]['message']}")
                        if 'details' in err['errors'][0]:
                            print(f"   Details: {err['errors'][0]['details']}")
                return None
            
            prompt_id = result.get('prompt_id')
            print(f'Queued: {prompt_id}')
        
        print('Generating video (may take 1-3 min on 3090)...')
        for i in range(180):  # 6 min timeout
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
                                print(f'\nSUCCESS!')
                                print(f'Video: {url}/view?filename={fn}')
                                return fn
                    if status.get('status_str') == 'error':
                        msgs = status.get('messages', [])
                        for m in msgs:
                            if m[0] == 'execution_error':
                                print(f"ERROR: {m[1].get('exception_message', 'Unknown')[:500]}")
                        return None
            if i % 15 == 0 and i > 0:
                print(f'   ... {i*2}s')
        
        print('Timeout')
        return None

if __name__ == '__main__':
    result = asyncio.run(generate_ltx_video())
    if result:
        print(f'\nDownload: http://192.168.1.143:8188/view?filename={result}')
