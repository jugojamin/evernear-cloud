"""Test full voice pipeline with REAL speech audio from macOS TTS.
Sends PCM frames exactly like the iOS app does.
"""
import asyncio, json, jwt, time, base64, subprocess, os, websockets

API_URL = "wss://evernear-api.fly.dev/ws/voice"
USER_ID = "5b1f2de9-8c66-4012-a2b8-480ef4257b38"
FRAME_SIZE = 4800  # bytes per frame (2400 samples * 2 bytes = 50ms at 48kHz)

def make_token():
    return jwt.encode({'sub': USER_ID, 'exp': int(time.time()) + 3600}, 'test-secret', algorithm='HS256')

def generate_speech_file(text: str, path: str):
    """Use macOS say + ffmpeg to generate PCM16 48kHz mono."""
    aiff = path + ".aiff"
    subprocess.run(["say", "-o", aiff, text], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", aiff,
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        path
    ], check=True, capture_output=True)
    os.remove(aiff)

def load_pcm_frames(path: str) -> list[str]:
    """Load PCM file and split into base64-encoded frames."""
    with open(path, "rb") as f:
        data = f.read()
    frames = []
    for i in range(0, len(data), FRAME_SIZE):
        chunk = data[i:i+FRAME_SIZE]
        if len(chunk) == FRAME_SIZE:
            frames.append(base64.b64encode(chunk).decode("ascii"))
    return frames

async def simulate_voice_turn(ws, turn_num: int, text: str):
    """Simulate one complete voice turn with real speech."""
    print(f"\n{'='*60}")
    print(f"TURN {turn_num}: Generating speech for: '{text}'")
    print(f"{'='*60}")
    
    # Generate speech audio
    pcm_path = f"/tmp/test_turn{turn_num}.pcm"
    generate_speech_file(text, pcm_path)
    frames = load_pcm_frames(pcm_path)
    print(f"  📢 Generated {len(frames)} audio frames from macOS TTS")
    
    # Send frames (simulates Hold-to-Talk)
    for frame_b64 in frames:
        await ws.send(json.dumps({"type": "audio", "data": frame_b64}))
        await asyncio.sleep(0.05)  # ~50ms per frame, matching real-time
    print(f"  📤 Sent {len(frames)} frames")
    
    # Button released
    await ws.send(json.dumps({"type": "end_of_speech"}))
    print(f"  ✋ end_of_speech sent")
    
    # Collect response
    got_transcript = False
    audio_count = 0
    response_text = ""
    
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=45)
            data = json.loads(msg)
            t = data.get("type", "")
            
            if t == "status":
                state = data.get("state", "")
                print(f"  📡 Status: {state}")
                if state == "listening" and (got_transcript or audio_count > 0):
                    print(f"  ✅ Turn {turn_num} COMPLETE — {audio_count} audio frames received")
                    return True
                    
            elif t == "transcript" and data.get("final"):
                got_transcript = True
                response_text = data.get("text", "")
                m = data.get("metrics", {})
                print(f"  💬 EverNear: \"{response_text[:120]}\"")
                print(f"     LLM: {m.get('llm_total_ms')}ms | Total: {m.get('total_ms')}ms | Model: {m.get('model_used')}")
                
            elif t == "audio":
                audio_count += 1
                is_last = data.get("last", False)
                if audio_count <= 2 or is_last:
                    print(f"  🔊 Audio frame {audio_count}{' (LAST)' if is_last else ''}")
                    
            elif t == "error":
                print(f"  ❌ Error: {data.get('message')}")
                return False
                
    except asyncio.TimeoutError:
        print(f"  ❌ Timed out. got_transcript={got_transcript}, audio_frames={audio_count}")
        return False

async def main():
    token = make_token()
    url = f"{API_URL}?token={token}"
    
    print("🎤 EverNear Full Voice Pipeline Test (Real Speech)")
    print("=" * 60)
    
    async with websockets.connect(url) as ws:
        print("✅ Connected to EverNear API")
        
        await ws.send(json.dumps({"type": "session_start"}))
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"✅ Session started\n")
        
        turns = [
            "Hello, how are you doing today?",
            "That's wonderful. Tell me about your day.",
            "You're such a good friend. Thank you.",
        ]
        
        results = []
        for i, text in enumerate(turns, 1):
            success = await simulate_voice_turn(ws, i, text)
            results.append(success)
            if not success:
                print(f"\n❌ FAILED at turn {i}")
                break
            await asyncio.sleep(2)
        
        await ws.send(json.dumps({"type": "session_end"}))
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {sum(results)}/{len(turns)} turns succeeded")
        for i, r in enumerate(results, 1):
            print(f"  Turn {i}: {'✅' if r else '❌'}")
        print(f"{'='*60}")

asyncio.run(main())
