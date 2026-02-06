"""Image-to-Video pipeline: Generate image then animate."""

import asyncio
import aiohttp
import uuid
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("./generated_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

COMFY_URL = "http://192.168.1.143:8188"

async def generate_image(session, prompt: str):
    """Generate a base image using FLUX."""
    print("\n[1/2] Generating base image...")
    
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, distorted, ugly", "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 832, "height": 480, "batch_size": 1}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 3.5,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0]
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "vibe_base"}
        }
    }
    
    client_id = str(uuid.uuid4())
    async with session.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": client_id}) as resp:
        result = await resp.json()
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        prompt_id = result["prompt_id"]
    
    print(f"    Prompt ID: {prompt_id}")
    
    while True:
        async with session.get(f"{COMFY_URL}/history/{prompt_id}") as resp:
            history = await resp.json()
        
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                for msg in status.get("messages", []):
                    if msg[0] == "execution_error":
                        print(f"    Error: {msg[1].get('exception_message')}")
                return None
            
            outputs = history[prompt_id].get("outputs", {})
            if outputs:
                for output in outputs.values():
                    if "images" in output:
                        img = output["images"][0]
                        filename = img["filename"]
                        subfolder = img.get("subfolder", "")
                        url = f"{COMFY_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                        print(f"    Generated: {filename}")
                        return url
        
        await asyncio.sleep(1)


async def animate_image(session, image_url: str, prompt: str):
    """Animate image using Wan I2V."""
    print("\n[2/2] Animating with Wan I2V...")
    
    # Download image first
    async with session.get(image_url) as resp:
        image_data = await resp.read()
    
    # Upload to ComfyUI
    form = aiohttp.FormData()
    form.add_field('image', image_data, filename='input.png', content_type='image/png')
    
    async with session.post(f"{COMFY_URL}/upload/image", data=form) as resp:
        upload_result = await resp.json()
        uploaded_name = upload_result.get("name", "input.png")
        print(f"    Uploaded: {uploaded_name}")
    
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"}
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_name}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt + ", smooth motion, cinematic", "clip": ["2", 0]}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "static, frozen, blurry, distorted", "clip": ["2", 0]}
        },
        "7": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 0],
                "width": 832,
                "height": 480,
                "length": 33,
                "batch_size": 1,
                "start_image": ["4", 0]
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2]
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]}
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "vibe_i2v",
                "format": "image/webp",
                "pingpong": False,
                "save_output": True
            }
        }
    }
    
    client_id = str(uuid.uuid4())
    async with session.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": client_id}) as resp:
        result = await resp.json()
        if "error" in result:
            print(f"Error: {result['error']}")
            print(f"Details: {result.get('node_errors', {})}")
            return None
        prompt_id = result["prompt_id"]
    
    print(f"    Prompt ID: {prompt_id}")
    print("    Generating video (2-5 minutes)...")
    
    while True:
        async with session.get(f"{COMFY_URL}/history/{prompt_id}") as resp:
            history = await resp.json()
        
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                for msg in status.get("messages", []):
                    if msg[0] == "execution_error":
                        print(f"    Error: {msg[1].get('exception_message')}")
                return None
            
            outputs = history[prompt_id].get("outputs", {})
            if outputs:
                for output in outputs.values():
                    if "gifs" in output:
                        gif = output["gifs"][0]
                        filename = gif["filename"]
                        subfolder = gif.get("subfolder", "")
                        url = f"{COMFY_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                        
                        async with session.get(url) as resp:
                            content = await resp.read()
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            local_path = OUTPUT_DIR / f"i2v_{timestamp}.webp"
                            local_path.write_bytes(content)
                            print(f"    Saved: {local_path}")
                            return local_path
        
        await asyncio.sleep(3)


async def main():
    print("=" * 60)
    print("IMAGE-TO-VIDEO PIPELINE")
    print("=" * 60)
    
    prompt = "cyberpunk samurai warrior standing in neon rain, cinematic"
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Generate image
        image_url = await generate_image(session, prompt)
        if not image_url:
            print("Failed to generate image")
            return
        
        # Step 2: Animate
        video_path = await animate_image(session, image_url, prompt)
        if video_path:
            print(f"\nComplete! Video: {video_path}")


if __name__ == "__main__":
    asyncio.run(main())
