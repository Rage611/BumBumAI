import asyncio
from groq import AsyncGroq
import config

async def test():
    cfg = config.load_config()
    key = cfg.get("GROQ_API_KEYS")[0]
    client = AsyncGroq(api_key=key)
    
    stream = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    print("Stream type:", type(stream))
    print("Stream dir:", dir(stream))
    print("Stream response dir:", dir(stream.response))
    print("Headers:")
    for k, v in stream.response.headers.items():
        if "ratelimit" in k.lower() or "limit" in k.lower() or "usage" in k.lower():
            print(f"  {k}: {v}")

asyncio.run(test())
