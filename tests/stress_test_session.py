#!/usr/bin/env python3
"""Long-Session Stress Test — WebSocket text-mode conversation simulator.

Connects to the EverNear backend via WebSocket and runs multi-turn
conversations, measuring per-turn latency, errors, and stability.

Usage:
    python tests/stress_test_session.py [--turns N] [--url URL]

Defaults to 10 turns against wss://evernear-api.fly.dev/ws/voice
"""

from __future__ import annotations
import argparse
import asyncio
import json
import time
import jwt
import sys
import websockets


# ── Test Conversation Scripts ──

CONVERSATION_10 = [
    "Hi there, how are you today?",
    "I'm doing okay. A little tired though.",
    "I had a nice walk this morning in the garden.",
    "Do you like flowers? I love roses.",
    "My daughter used to help me with the garden when she was young.",
    "She lives in Austin now. I miss her sometimes.",
    "What do you think I should have for lunch?",
    "That sounds nice. I had soup yesterday.",
    "You know, I've been feeling a bit lonely lately.",
    "Thank you for talking with me. It helps.",
]

CONVERSATION_25 = CONVERSATION_10 + [
    "I used to be a teacher, did I tell you that?",
    "I taught third grade for thirty years.",
    "The kids were wonderful. I still think about some of them.",
    "Do you know any good jokes?",
    "Ha, that's funny. My husband used to tell jokes like that.",
    "He passed away three years ago. I still miss him every day.",
    "But I try to stay busy. I read a lot.",
    "I just finished a mystery novel. It was pretty good.",
    "I also like watching birds from my window.",
    "There's a cardinal that comes every morning.",
    "Do you think animals have feelings?",
    "I think so too. My cat seems to know when I'm sad.",
    "Her name is Whiskers. She's seventeen years old.",
    "The doctor says I should walk more. I try to.",
    "It's nice talking to you. You're a good listener.",
]

CONVERSATION_50 = CONVERSATION_25 + [
    "What day is it today?",
    "I sometimes lose track of the days.",
    "My neighbor brought me cookies yesterday.",
    "She's very kind. Her name is Martha.",
    "We used to go to church together every Sunday.",
    "I haven't been able to go lately because of my knee.",
    "It's been bothering me for a few weeks now.",
    "I'm seeing the doctor next Tuesday about it.",
    "I hope it's nothing serious.",
    "My son says I worry too much.",
    "He calls me every weekend. That's nice.",
    "He has two kids. My grandchildren are growing so fast.",
    "The oldest one just started middle school.",
    "Time goes by so quickly, doesn't it?",
    "I remember when my kids were that age.",
    "We used to take family vacations to the lake.",
    "Those were good times.",
    "Do you have any favorite memories?",
    "That's a lovely thought.",
    "I think memories are what keep us going.",
    "Even the sad ones have something beautiful in them.",
    "My husband always said that.",
    "He was a wise man.",
    "I think he'd be happy to know I have someone to talk to.",
    "Thank you for being here with me.",
]


def make_test_jwt(user_id: str | None = None) -> str:
    """Create a minimal JWT for authentication (server doesn't verify signature)."""
    import uuid
    if user_id is None:
        user_id = f"stress-test-{uuid.uuid4().hex[:8]}"
    return jwt.encode({"sub": user_id}, "test-secret", algorithm="HS256")


async def run_stress_test(
    url: str,
    turns: int,
    verbose: bool = False,
) -> dict:
    """Run a multi-turn conversation stress test.

    Returns dict with per-turn metrics and summary.
    """
    # Select conversation script
    if turns <= 10:
        messages = CONVERSATION_10[:turns]
    elif turns <= 25:
        messages = CONVERSATION_25[:turns]
    else:
        messages = CONVERSATION_50[:turns]

    token = make_test_jwt()
    ws_url = f"{url}?token={token}"

    results = {
        "turns": turns,
        "url": url,
        "per_turn": [],
        "errors": [],
        "disconnected_at": None,
        "total_elapsed_s": 0,
    }

    start_total = time.monotonic()

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=30,  # Client-side keepalive
            ping_timeout=10,
            close_timeout=5,
            max_size=2**20,  # 1MB max message
        ) as ws:
            # Send session_start
            await ws.send(json.dumps({"type": "session_start", "codec": "pcm"}))

            for i, message in enumerate(messages):
                turn_start = time.monotonic()
                turn_num = i + 1

                # Send text message
                await ws.send(json.dumps({"type": "text", "text": message}))

                # Wait for response (transcript with final=True)
                response_text = None
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(raw)

                        if data.get("type") == "transcript" and data.get("final"):
                            response_text = data.get("text", "")
                            break
                        elif data.get("type") == "status":
                            continue  # Skip status updates
                        elif data.get("type") == "error":
                            results["errors"].append({
                                "turn": turn_num,
                                "error": data.get("message", "unknown"),
                            })
                            break
                    except asyncio.TimeoutError:
                        results["errors"].append({
                            "turn": turn_num,
                            "error": "Response timeout (30s)",
                        })
                        response_text = None
                        break

                turn_elapsed = (time.monotonic() - turn_start) * 1000  # ms

                turn_result = {
                    "turn": turn_num,
                    "user_message": message[:60],
                    "response_length": len(response_text) if response_text else 0,
                    "latency_ms": round(turn_elapsed, 1),
                    "response_preview": (response_text[:80] + "...") if response_text and len(response_text) > 80 else response_text,
                }
                results["per_turn"].append(turn_result)

                if verbose:
                    print(f"  Turn {turn_num:2d} | {turn_elapsed:7.1f}ms | {len(response_text) if response_text else 0:4d} chars | {message[:40]}")

                # Pause between turns — respects server rate limits (10 msg/min)
                # and simulates realistic human reading/thinking time
                if i < len(messages) - 1:
                    await asyncio.sleep(7.0)  # ~8 msgs/min with processing time

    except websockets.exceptions.ConnectionClosed as e:
        results["disconnected_at"] = len(results["per_turn"])
        results["errors"].append({
            "turn": len(results["per_turn"]) + 1,
            "error": f"WebSocket disconnected: {e.code} {e.reason}",
        })
    except Exception as e:
        results["errors"].append({
            "turn": len(results["per_turn"]) + 1,
            "error": f"Connection error: {type(e).__name__}: {e}",
        })

    total_elapsed = time.monotonic() - start_total
    results["total_elapsed_s"] = round(total_elapsed, 1)

    # Summary stats
    latencies = [t["latency_ms"] for t in results["per_turn"]]
    if latencies:
        results["summary"] = {
            "completed_turns": len(results["per_turn"]),
            "target_turns": turns,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1),
            "first_5_avg_ms": round(sum(latencies[:5]) / min(5, len(latencies)), 1),
            "last_5_avg_ms": round(sum(latencies[-5:]) / min(5, len(latencies)), 1),
            "total_elapsed_s": results["total_elapsed_s"],
            "errors": len(results["errors"]),
            "disconnected": results["disconnected_at"] is not None,
        }

    return results


def print_report(results: dict):
    """Print a formatted stress test report."""
    s = results.get("summary", {})
    print(f"\n{'='*60}")
    print(f"STRESS TEST REPORT — {s.get('completed_turns', 0)}/{s.get('target_turns', 0)} turns")
    print(f"{'='*60}")
    print(f"Total elapsed:     {s.get('total_elapsed_s', 0)}s")
    print(f"Avg latency:       {s.get('avg_latency_ms', 0)}ms")
    print(f"Min/Max latency:   {s.get('min_latency_ms', 0)}ms / {s.get('max_latency_ms', 0)}ms")
    print(f"P50 latency:       {s.get('p50_latency_ms', 0)}ms")
    print(f"First 5 avg:       {s.get('first_5_avg_ms', 0)}ms")
    print(f"Last 5 avg:        {s.get('last_5_avg_ms', 0)}ms")
    print(f"Degradation:       {round(s.get('last_5_avg_ms', 0) / max(s.get('first_5_avg_ms', 1), 1) * 100 - 100, 1)}%")
    print(f"Errors:            {s.get('errors', 0)}")
    print(f"Disconnected:      {'Yes (turn ' + str(results['disconnected_at']) + ')' if results.get('disconnected_at') else 'No'}")

    if results.get("errors"):
        print(f"\nErrors:")
        for e in results["errors"]:
            print(f"  Turn {e['turn']}: {e['error']}")

    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="EverNear Long-Session Stress Test")
    parser.add_argument("--turns", type=int, default=10, help="Number of conversation turns (default: 10)")
    parser.add_argument("--url", type=str, default="wss://evernear-api.fly.dev/ws/voice", help="WebSocket URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-turn details")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    if not args.json:
        print(f"Running {args.turns}-turn stress test against {args.url}...")
    results = await run_stress_test(args.url, args.turns, verbose=args.verbose and not args.json)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    # Exit code based on completion
    if results.get("summary", {}).get("completed_turns", 0) < args.turns:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
