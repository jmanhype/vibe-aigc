"""Test Zhipu CogView image generation."""
import asyncio
import httpx
import os
import json

async def test_cogview():
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Try different Zhipu image endpoints/models
    models_to_try = ["cogview-3-plus", "cogview-3", "cogview-4"]
    
    for model in models_to_try:
        print(f"\nTrying model: {model}")
        
        url = "https://open.bigmodel.cn/api/paas/v4/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "prompt": "A chrome android with glowing blue eyes and purple hair, cyberpunk neon noir style"
        }
        
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(url, json=data, headers=headers)
                print(f"Status: {r.status_code}")
                
                if r.status_code == 200:
                    result = r.json()
                    print(f"SUCCESS!")
                    if "data" in result and len(result["data"]) > 0:
                        img_url = result["data"][0].get("url", "No URL")
                        print(f"Image URL: {img_url}")
                        return img_url
                else:
                    # Print error but handle encoding
                    try:
                        err = r.json()
                        print(f"Error: {json.dumps(err, ensure_ascii=True)}")
                    except:
                        print(f"Error: {r.status_code}")
        except Exception as e:
            print(f"Exception: {e}")
    
    return None

if __name__ == "__main__":
    asyncio.run(test_cogview())
