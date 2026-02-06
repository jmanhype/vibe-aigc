import asyncio, aiohttp

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get("http://192.168.1.143:8188/history/beb815da-3335-4c91-8d8a-16802b2c6c94") as r:
            h = await r.json()
        
        pid = "beb815da-3335-4c91-8d8a-16802b2c6c94"
        if pid in h:
            status = h[pid].get("status", {})
            print("Status:", status.get("status_str", "unknown"))
            
            if status.get("status_str") == "error":
                for msg in status.get("messages", []):
                    if msg[0] == "execution_error":
                        print("Error:", msg[1].get("exception_message", "")[:300])
            
            outputs = h[pid].get("outputs", {})
            if outputs:
                print("HAS OUTPUTS!")
                for k, v in outputs.items():
                    print(f"  Node {k}: {list(v.keys())}")
        else:
            print("Job not found in history")

asyncio.run(main())
