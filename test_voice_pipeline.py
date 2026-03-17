"""Quick test of the EverNear voice pipeline via WebSocket text mode."""
import asyncio
import json
import websockets

API_URL = "wss://evernear-api.fly.dev/ws/voice"
USER_ID = "5b1f2de9-8c66-4012-a2b8-480ef4257b38"

def make_token():
    import jwt, time
    return jwt.encode({'sub': USER_ID, 'exp': int(time.time()) + 3600}, 'test-secret', algorithm='HS256')

async def test():
    token = make_token()
    url = f"{API_URL}?token={token}"
    print(f"Connecting to {url}...")
    
    async with websockets.connect(url) as ws:
        print("Connected!")
        
        # Start session
        await ws.send(json.dumps({"type": "session_start"}))
        print("Sent session_start")
        
        # Wait for listening status
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"Received: {msg}")
        
        # Send text message (bypasses STT entirely)
        await ws.send(json.dumps({"type": "text", "text": "Hello, how are you today?"}))
        print("Sent text message: 'Hello, how are you today?'")
        
        # Collect responses
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=45)
                data = json.loads(msg) if isinstance(msg, str) else {"type": "binary", "len": len(msg)}
                print(f"Received: {json.dumps(data, indent=2)[:500]}")
                
                # If we got a final transcript back, we're done
                if isinstance(data, dict) and data.get("type") == "transcript" and data.get("final"):
                    print("\n✅ Got final response!")
                    break
                if isinstance(data, dict) and data.get("type") == "status" and data.get("state") == "listening":
                    print("\n✅ Back to listening — turn complete!")
                    break
        except asyncio.TimeoutError:
            print("\n❌ Timed out waiting for response (45s)")
        
        # End session
        await ws.send(json.dumps({"type": "session_end"}))
        print("Session ended.")

asyncio.run(test())
