"""Latency benchmark — 10 text turns against live EverNear API."""
import asyncio
import json
import time
import statistics
import ssl
import websockets

WS = "wss://evernear-api.fly.dev/ws/voice"
# Server decodes JWT without signature verification, just needs 'sub' claim
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDEiLCJyb2xlIjoiYXV0aGVudGljYXRlZCJ9.wFGkRI0z-fjRq3cvtErj34JVx-1bdoHeRYRwICGhmmM"

PROMPTS = [
    "Good morning!",
    "How are you today?",
    "I took my medication this morning.",
    "My daughter Sarah called me yesterday.",
    "We talked about her kids for a while.",
    "I've been feeling a little tired lately.",
    "What's a good way to stay active?",
    "I used to love gardening when I was younger.",
    "Do you remember my daughter's name?",
    "Thank you for talking with me.",
]


async def run_benchmark():
    ssl_ctx = ssl.create_default_context()
    results = []

    try:
        async with websockets.connect(
            f"{WS}?token={TOKEN}",
            ssl=ssl_ctx,
        ) as ws:
            print("Connected to WebSocket")
            
            # Start session
            await ws.send(json.dumps({"type": "session_start"}))
            # Drain initial messages
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3)
                    d = json.loads(msg)
                    print(f"  Init: {d.get('type', 'unknown')}")
                    if d.get("type") != "status":
                        break
            except asyncio.TimeoutError:
                print("  (no more init messages)")

            for i, prompt in enumerate(PROMPTS):
                start = time.perf_counter()

                await ws.send(json.dumps({
                    "type": "text",
                    "text": prompt,
                }))

                # Collect response (skip status messages)
                response_data = None
                try:
                    while True:
                        resp = await asyncio.wait_for(ws.recv(), timeout=30)
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        data = json.loads(resp)
                        if data.get("type") == "status":
                            continue
                        response_data = data
                        break
                except asyncio.TimeoutError:
                    elapsed_ms = 30000

                if response_data:
                    response_text = response_data.get("text", "")[:80]
                    metrics = response_data.get("metrics", {})
                    total = round(elapsed_ms)

                    results.append({
                        "turn": i + 1,
                        "prompt": prompt,
                        "total_ms": total,
                        "response": response_text,
                        "server_metrics": metrics,
                    })
                    print(f"  Turn {i+1}: {total}ms — \"{response_text}\"")
                else:
                    results.append({"turn": i + 1, "prompt": prompt, "total_ms": 30000, "response": "TIMEOUT"})
                    print(f"  Turn {i+1}: TIMEOUT")

                # Drain trailing status
                try:
                    while True:
                        extra = await asyncio.wait_for(ws.recv(), timeout=1)
                        extra_data = json.loads(extra)
                        if extra_data.get("type") == "status":
                            continue
                        break
                except asyncio.TimeoutError:
                    pass

            await ws.send(json.dumps({"type": "session_end"}))

    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    times = [r["total_ms"] for r in results if r["total_ms"] < 30000]
    if times:
        p95_idx = min(int(len(times) * 0.95), len(times) - 1)
        print(f"\n{'='*60}")
        print(f"LATENCY BENCHMARK — {len(times)} turns")
        print(f"{'='*60}")
        print(f"  Median:  {round(statistics.median(times))}ms")
        print(f"  Mean:    {round(statistics.mean(times))}ms")
        print(f"  P95:     {round(sorted(times)[p95_idx])}ms")
        print(f"  Min:     {min(times)}ms")
        print(f"  Max:     {max(times)}ms")
        print(f"  Target:  <1000ms")
        print(f"  Result:  {'✅ PASS' if statistics.median(times) < 1000 else '❌ MISS'}")

        print(f"\nPer-turn breakdown:")
        for r in results:
            sm = r.get("server_metrics", {})
            stt = sm.get("stt_ms", "—")
            llm_ttft = sm.get("llm_ttft_ms", "—")
            llm_total = sm.get("llm_total_ms", "—")
            tts = sm.get("tts_ttfb_ms", "—")
            model = sm.get("model_used", "—")
            print(f"  Turn {r['turn']:2d}: {r['total_ms']:5d}ms  stt:{stt}  llm_ttft:{llm_ttft}  llm:{llm_total}  tts:{tts}  model:{model}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
