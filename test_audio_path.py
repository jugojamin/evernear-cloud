"""Simulate iPhone Hold-to-Talk: send real audio frames via WebSocket.
Tests the full voice path: Deepgram STT → Claude → Cartesia TTS → audio back.
"""
import asyncio, json, jwt, time, struct, math, websockets

API_URL = "wss://evernear-api.fly.dev/ws/voice"
USER_ID = "5b1f2de9-8c66-4012-a2b8-480ef4257b38"

def make_token():
    return jwt.encode({'sub': USER_ID, 'exp': int(time.time()) + 3600}, 'test-secret', algorithm='HS256')

def generate_speech_pcm(text_hint: str, duration_s: float = 2.0, sample_rate: int = 48000) -> list[bytes]:
    """Generate PCM16 audio frames that Deepgram can transcribe.
    
    Since we can't generate real speech easily, we'll use a different approach:
    send silence frames to establish the connection, then use end_of_speech.
    
    For actual STT testing, we need real speech audio.
    Let's generate a simple tone that at least proves the audio pipeline works.
    """
    samples_per_frame = 2400  # 50ms at 48kHz (matches iOS app frame size)
    total_samples = int(sample_rate * duration_s)
    frames = []
    
    # Generate a 440Hz tone (won't transcribe as speech, but tests the full pipeline)
    for start in range(0, total_samples, samples_per_frame):
        end = min(start + samples_per_frame, total_samples)
        samples = []
        for i in range(start, end):
            t = i / sample_rate
            # 440Hz sine wave at ~50% amplitude
            sample = int(16000 * math.sin(2 * math.pi * 440 * t))
            samples.append(max(-32768, min(32767, sample)))
        frame_bytes = struct.pack(f'<{len(samples)}h', *samples)
        frames.append(frame_bytes)
    
    return frames

def pcm_to_base64(pcm_bytes: bytes) -> str:
    import base64
    return base64.b64encode(pcm_bytes).decode('ascii')

async def simulate_turn(ws, turn_num: int, duration_s: float = 2.0):
    """Simulate one Hold-to-Talk turn."""
    print(f"\n{'='*50}")
    print(f"TURN {turn_num}: Simulating {duration_s}s button press")
    print(f"{'='*50}")
    
    # Generate audio frames (like iOS AudioCaptureService)
    frames = generate_speech_pcm(f"turn {turn_num}", duration_s)
    print(f"  Generated {len(frames)} audio frames ({len(frames[0])} bytes each)")
    
    # Send audio frames (simulates Hold-to-Talk held down)
    for i, frame in enumerate(frames):
        await ws.send(json.dumps({
            "type": "audio",
            "data": pcm_to_base64(frame),
        }))
    print(f"  Sent {len(frames)} frames to server")
    
    # Button released → end_of_speech (simulates finger lifted)
    await ws.send(json.dumps({"type": "end_of_speech"}))
    print(f"  Sent end_of_speech")
    
    # Wait for response
    got_transcript = False
    got_audio = False
    audio_frame_count = 0
    response_text = ""
    
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=45)
            data = json.loads(msg)
            msg_type = data.get("type", "")
            
            if msg_type == "status":
                state = data.get("state", "")
                print(f"  Status: {state}")
                if state == "listening" and (got_transcript or got_audio):
                    print(f"  ✅ Turn {turn_num} complete! Audio frames received: {audio_frame_count}")
                    return True
                    
            elif msg_type == "transcript":
                response_text = data.get("text", "")
                is_final = data.get("final", False)
                metrics = data.get("metrics", {})
                if is_final:
                    got_transcript = True
                    print(f"  📝 Response: {response_text[:100]}...")
                    print(f"     LLM: {metrics.get('llm_total_ms')}ms | Total: {metrics.get('total_ms')}ms")
                    
            elif msg_type == "audio":
                audio_frame_count += 1
                got_audio = True
                if audio_frame_count <= 2 or audio_frame_count % 10 == 0:
                    print(f"  🔊 Audio frame {audio_frame_count} ({len(data.get('data',''))} chars b64)")
                    
            elif msg_type == "error":
                print(f"  ❌ Error: {data.get('message', 'unknown')}")
                return False
                
    except asyncio.TimeoutError:
        print(f"  ❌ Turn {turn_num} timed out (45s)")
        if got_transcript:
            print(f"     (Got transcript but no audio/listening status)")
        return False

async def main():
    token = make_token()
    url = f"{API_URL}?token={token}"
    
    print(f"Connecting to EverNear API...")
    print(f"NOTE: Sending tone audio (not speech). Deepgram may not produce")
    print(f"a transcript, but this tests the full server pipeline path.")
    print(f"If Deepgram can't transcribe, the failure script should fire.\n")
    
    async with websockets.connect(url) as ws:
        print("✅ WebSocket connected")
        
        # Start session
        await ws.send(json.dumps({"type": "session_start"}))
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"Session started: {msg}")
        
        # Test 3 turns
        for turn in range(1, 4):
            success = await simulate_turn(ws, turn, duration_s=2.0)
            if not success:
                print(f"\n❌ Failed at turn {turn}")
                break
            await asyncio.sleep(2)  # Pause between turns
        else:
            print(f"\n🎉 All 3 turns completed!")
        
        await ws.send(json.dumps({"type": "session_end"}))
        print("\nSession ended.")

asyncio.run(main())
