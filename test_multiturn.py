"""Test multi-turn conversation via WebSocket text mode."""
import asyncio, json, jwt, time, websockets

API_URL = "wss://evernear-api.fly.dev/ws/voice"
USER_ID = "5b1f2de9-8c66-4012-a2b8-480ef4257b38"

def make_token():
    return jwt.encode({'sub': USER_ID, 'exp': int(time.time()) + 3600}, 'test-secret', algorithm='HS256')

async def send_and_wait(ws, text, turn_num):
    print(f"\n--- Turn {turn_num}: '{text}' ---")
    await ws.send(json.dumps({"type": "text", "text": text}))
    
    got_response = False
    audio_count = 0
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=45)
            if isinstance(msg, bytes):
                audio_count += 1
                continue
            data = json.loads(msg)
            if data.get("type") == "audio":
                audio_count += 1
            elif data.get("type") == "transcript" and data.get("final"):
                print(f"  Response: {data['text'][:120]}...")
                print(f"  LLM: {data.get('metrics',{}).get('llm_total_ms')}ms | Total: {data.get('metrics',{}).get('total_ms')}ms")
                got_response = True
            elif data.get("type") == "status" and data.get("state") == "listening":
                if got_response:
                    print(f"  Audio frames: {audio_count}")
                    print(f"  ✅ Turn {turn_num} complete!")
                    return True
    except asyncio.TimeoutError:
        print(f"  ❌ Turn {turn_num} timed out!")
        return False

async def test():
    token = make_token()
    url = f"{API_URL}?token={token}"
    
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "session_start"}))
        await asyncio.wait_for(ws.recv(), timeout=5)  # listening status
        
        turns = [
            "Hello, how are you today?",
            "That's great. What's your favorite thing to talk about?",
            "Tell me something interesting.",
        ]
        
        for i, text in enumerate(turns, 1):
            success = await send_and_wait(ws, text, i)
            if not success:
                print(f"\n❌ Failed at turn {i}")
                break
            await asyncio.sleep(1)  # Brief pause between turns
        else:
            print(f"\n🎉 All {len(turns)} turns completed successfully!")
        
        await ws.send(json.dumps({"type": "session_end"}))

asyncio.run(test())
