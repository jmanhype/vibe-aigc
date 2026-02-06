"""Direct Wan 2.1 video generation test."""

import asyncio
import aiohttp
import uuid
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("./generated_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_wan_t2v_workflow(prompt: str, negative: str = "", width: int = 832, height: int = 480, frames: int = 81):
    """Create Wan 2.1 Text-to-Video workflow."""
    return {
        # Load Wan model
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.1_t2v_1.3b_bf16.safetensors",
                "weight_dtype": "default"
            }
        },
        # Load CLIP (umt5)
        "2": {
            "class_type": "CLIPLoader", 
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan"
            }
        },
        # Load VAE
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            }
        },
        # Encode positive
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 0]
            }
        },
        # Encode negative
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative or "blurry, low quality, distorted, ugly",
                "clip": ["2", 0]
            }
        },
        # Empty latent
        "6": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1
            }
        },
        # KSampler
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 30,
                "cfg": 6.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0]
            }
        },
        # VAE decode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0]
            }
        },
        # Save video
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "vibe_wan",
                "format": "image/webp",
                "pingpong": False,
                "save_output": True
            }
        }
    }


async def main():
    print("=" * 60)
    print("WAN 2.1 VIDEO GENERATION")
    print("=" * 60)
    print()
    
    prompt = "cyberpunk samurai walking through neon rain, cinematic, slow motion"
    
    workflow = create_wan_t2v_workflow(
        prompt=prompt,
        width=832,
        height=480,
        frames=81  # ~5 sec at 16fps
    )
    
    print(f"Prompt: {prompt}")
    print("Submitting to ComfyUI...")
    
    async with aiohttp.ClientSession() as session:
        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        
        async with session.post(
            "http://192.168.1.143:8188/prompt",
            json=payload
        ) as resp:
            result = await resp.json()
            
            if "error" in result:
                print(f"Error: {result['error']}")
                print(f"Details: {result.get('node_errors', {})}")
                return None
            
            prompt_id = result.get("prompt_id")
            print(f"Prompt ID: {prompt_id}")
        
        print("Generating video (this takes 2-5 minutes)...")
        
        while True:
            async with session.get(f"http://192.168.1.143:8188/history/{prompt_id}") as resp:
                history = await resp.json()
            
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                status_str = status.get("status_str", "")
                
                if status_str == "error":
                    for msg in status.get("messages", []):
                        if msg[0] == "execution_error":
                            print(f"Error: {msg[1].get('exception_message', 'Unknown error')}")
                    return None
                
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    for node_id, output in outputs.items():
                        if "gifs" in output:
                            for gif in output["gifs"]:
                                filename = gif.get("filename")
                                subfolder = gif.get("subfolder", "")
                                
                                url = f"http://192.168.1.143:8188/view?filename={filename}&subfolder={subfolder}&type=output"
                                print(f"Downloading: {url}")
                                
                                async with session.get(url) as resp:
                                    content = await resp.read()
                                    
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    local_path = OUTPUT_DIR / f"wan_{timestamp}.webp"
                                    local_path.write_bytes(content)
                                    
                                    print(f"Saved: {local_path}")
                                    return local_path
                    
                    print(f"Outputs but no video: {list(outputs.keys())}")
                    return None
            
            await asyncio.sleep(3)


if __name__ == "__main__":
    path = asyncio.run(main())
    if path:
        print(f"\nVideo saved: {path}")
