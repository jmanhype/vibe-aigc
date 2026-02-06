import asyncio, aiohttp

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get("http://192.168.1.143:8188/queue") as r:
            q = await r.json()
        
        running = q.get("queue_running", [])
        print(f"Running: {len(running)}")
        
        for job in running:
            print(f"  ID: {job[1][:20]}...")
            # Check if it's our job
            if "beb815da" in job[1]:
                print("  ^ This is our I2V job")

asyncio.run(main())
