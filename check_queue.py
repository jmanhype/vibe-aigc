import asyncio, aiohttp

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get("http://192.168.1.143:8188/queue") as r:
            q = await r.json()
        print("Running:", len(q.get("queue_running", [])))

asyncio.run(main())
