import asyncio
import aiohttp
import random

async def generate_wan_video():
    url = 'http://192.168.1.143:8188'
    prompt = 'cyberpunk samurai walking through neon tokyo rain, cinematic, blade runner'
    seed = random.randint(0, 2**32)
    
    # Wan 2.2 Video workflow (text-to-video)
    # Using AniSoraV3_2 HIGH quality model
    workflow = {
        # Load Wan model
        '1': {'class_type': 'UNETLoader', 
              'inputs': {
                  'unet_name': 'I2V/AniSora/Wan2_2-I2V_AniSoraV3_2_HIGH_14B_fp8_e4m3fn_scaled_KJ.safetensors',
                  'weight_dtype': 'fp8_e4m3fn'
              }},
        # Load text encoder
        '2': {'class_type': 'CLIPLoaderGGUF',
              'inputs': {
                  'clip_name': 'umt5_xxl_fp8_e4m3fn_scaled.safetensors',
                  'type': 'wan'
              }},
        # Load VAE
        '3': {'class_type': 'VAELoader',
              'inputs': {'vae_name': 'wan2.2_vae.safetensors'}},
        
        # Encode prompts
        '4': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': prompt, 'clip': ['2', 0]}},
        '5': {'class_type': 'CLIPTextEncode', 
              'inputs': {'text': 'blurry, static, low quality, watermark', 'clip': ['2', 0]}},
        
        # Empty latent (for T2V we need to create empty latent)
        '6': {'class_type': 'EmptySD3LatentImage',
              'inputs': {'width': 512, 'height': 320, 'batch_size': 1}},
        
        # KSampler
        '7': {'class_type': 'KSampler',
              'inputs': {
                  'model': ['1', 0],
                  'positive': ['4', 0],
                  'negative': ['5', 0],
                  'latent_image': ['6', 0],
                  'seed': seed,
                  'steps': 20,
                  'cfg': 6.0,
                  'sampler_name': 'euler',
                  'scheduler': 'normal',
                  'denoise': 1.0
              }},
        
        # Decode and save
        '8': {'class_type': 'VAEDecode',
              'inputs': {'samples': ['7', 0], 'vae': ['3', 0]}},
        '9': {'class_type': 'SaveImage',
              'inputs': {'images': ['8', 0], 'filename_prefix': 'wan_3090'}}
    }
    
    async with aiohttp.ClientSession() as session:
        print('>>> Queuing Wan 2.2 AniSora on 3090...')
        print(f'    Prompt: {prompt}')
        async with session.post(f'{url}/prompt', json={'prompt': workflow}) as resp:
            result = await resp.json()
            
            if 'error' in result:
                print(f"ERROR: {result['error']['message']}")
                if 'node_errors' in result:
                    for nid, err in result['node_errors'].items():
                        print(f"   Node {nid}: {err['errors'][0]['message']}")
                        if 'details' in err['errors'][0]:
                            print(f"   Details: {err['errors'][0]['details'][:200]}")
                return None
            
            prompt_id = result.get('prompt_id')
            print(f'Queued: {prompt_id}')
        
        print('Generating (may take 1-2 min)...')
        for i in range(120):
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
                                print(f'Image: {url}/view?filename={fn}')
                                return fn
                    if status.get('status_str') == 'error':
                        msgs = status.get('messages', [])
                        for m in msgs:
                            if m[0] == 'execution_error':
                                print(f"ERROR: {m[1].get('exception_message', 'Unknown')[:300]}")
                        return None
            if i % 15 == 0 and i > 0:
                print(f'   ... {i*2}s')
        
        print('Timeout')
        return None

if __name__ == '__main__':
    result = asyncio.run(generate_wan_video())
