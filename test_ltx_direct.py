"""Direct LTX video generation test."""

import asyncio
import aiohttp
import json
import uuid
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("./generated_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_ltx_workflow(prompt: str, negative: str = "", width: int = 512, height: int = 320, frames: int = 49):
    """Create LTX-2 Video workflow."""
    return {
        # Load model (MODEL output at index 0)
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "ltx-2-19b-distilled-fp8.safetensors"
            }
        },
        # Load text encoder for LTX-2 (CLIP output at index 0)
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": "ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
                "ckpt_name": "ltx-2-19b-distilled-fp8.safetensors",
                "device": "default"
            }
        },
        # VAE
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "LTX2_video_vae_bf16.safetensors"
            }
        },
        # Encode positive prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 0]  # CLIP from LTXAVTextEncoderLoader
            }
        },
        # Encode negative prompt
        "5": {
            "class_type": "CLIPTextEncode", 
            "inputs": {
                "text": negative or "blurry, low quality, distorted",
                "clip": ["2", 0]
            }
        },
        # LTX conditioning
        "6": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "frame_rate": 24.0
            }
        },
        # Empty latent
        "7": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1
            }
        },
        # Scheduler
        "8": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": 20,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["7", 0]
            }
        },
        # Sampler select
        "10": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        # Sample
        "9": {
            "class_type": "SamplerCustom",
            "inputs": {
                "add_noise": True,
                "noise_seed": 42,
                "cfg": 3.0,
                "model": ["1", 0],  # MODEL from CheckpointLoaderSimple
                "positive": ["6", 0],
                "negative": ["6", 1],
                "sampler": ["10", 0],
                "sigmas": ["8", 0],
                "latent_image": ["7", 0]
            }
        },
        # VAE decode
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0]
            }
        },
        # Save video
        "12": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["11", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "vibe_ltx",
                "format": "image/webp",
                "pingpong": False,
                "save_output": True
            }
        }
    }


async def main():
    print("=" * 60)
    print("LTX-2 VIDEO GENERATION")
    print("=" * 60)
    print()
    
    prompt = "cyberpunk samurai walking through neon rain, cinematic"
    
    workflow = create_ltx_workflow(
        prompt=prompt,
        width=512,
        height=320,
        frames=49
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
        
        print("Generating video...")
        
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
                                    local_path = OUTPUT_DIR / f"ltx_{timestamp}.webp"
                                    local_path.write_bytes(content)
                                    
                                    print(f"Saved: {local_path}")
                                    return local_path
                    
                    print(f"Outputs but no video: {outputs}")
                    return None
            
            await asyncio.sleep(2)


if __name__ == "__main__":
    path = asyncio.run(main())
    if path:
        print(f"\nVideo saved: {path}")
