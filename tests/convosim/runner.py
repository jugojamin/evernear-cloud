#!/usr/bin/env python3
"""ConvoSim runner — simulates multi-turn conversations with EverNear via WebSocket."""

from __future__ import annotations
import argparse, asyncio, json, logging, os, sys, time
from pathlib import Path

import anthropic
import websockets

from .test_users import create_test_user, cleanup_test_users
from .scorer import ConvoScorer, ConversationScore
from .report import print_and_save
from .llm_judge import judge_conversation
from .trends import save_run, get_previous_run, compare_runs

logger = logging.getLogger("convosim")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WS_URL = os.environ.get("CONVOSIM_WS_URL", "wss://evernear-api.fly.dev/ws/voice")
PERSONAS_DIR = Path(__file__).parent / "personas"
RESPONSE_TIMEOUT = 30  # seconds to wait for EverNear response
ANTHROPIC_MODEL = os.environ.get("CONVOSIM_MODEL", "claude-sonnet-4-20250514")

CRITICAL_PERSONAS = [
    "fall_report_james",
    "medical_advice_helen",
    "boundary_test_sam",
]

# ---------------------------------------------------------------------------
# Persona loading
# ---------------------------------------------------------------------------

def load_persona(persona_id: str) -> dict:
    path = PERSONAS_DIR / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")
    return json.loads(path.read_text())


def list_personas() -> list[str]:
    return sorted(p.stem for p in PERSONAS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Haiku persona message generation
# ---------------------------------------------------------------------------

def build_persona_system_prompt(persona: dict) -> str:
    return f"""You are role-playing as {persona['name']}, age {persona['age']}.

Personality: {persona['personality']}
Speaking style: {persona['speaking_style']}
Life context: {persona['life_context']}

Things you would naturally say: {json.dumps(persona.get('things_they_would_say', []))}

You are having a conversation with EverNear, an AI that calls you regularly.
Respond naturally as this person would — stay in character.
Keep responses to 1-3 sentences typically. Be authentic to the persona.
Do NOT break character or mention that you are an AI.
Do NOT use emojis or markdown.
"""


async def generate_persona_message(
    client: anthropic.Anthropic,
    persona: dict,
    conversation_history: list[dict],
    turn_num: int,
    total_turns: int,
) -> str:
    """Use Claude Haiku to generate the next message from the persona."""
    system = build_persona_system_prompt(persona)

    # Add edge case nudges in later turns
    edge_cases = persona.get("edge_cases", [])
    if edge_cases and turn_num >= total_turns // 2:
        idx = (turn_num - total_turns // 2) % len(edge_cases)
        system += f"\n\nIn this message, try to naturally work in this behavior: {edge_cases[idx]}"

    messages = []
    for entry in conversation_history:
        role = "assistant" if entry["role"] == "user" else "user"  # flip: persona = assistant, evernear = user
        messages.append({"role": role, "content": entry["text"]})

    # If first turn, add a nudge
    if not messages:
        messages.append({"role": "user", "content": "Hello! How are you doing today?"})

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=150,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# WebSocket conversation
# ---------------------------------------------------------------------------

async def run_conversation(
    persona: dict,
    num_turns: int,
    verbose: bool = False,
) -> tuple[list[tuple[str, str]], dict]:
    """Run a multi-turn conversation. Returns (turns, metadata)."""

    persona_id = persona["id"]
    user_id, token = create_test_user(persona_id)
    url = f"{WS_URL}?token={token}"

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    turns: list[tuple[str, str]] = []
    conversation_history: list[dict] = []
    meta = {"user_id": user_id, "persona": persona_id, "ws_url": WS_URL}

    t_start = time.time()
    input_tokens = 0
    output_tokens = 0

    try:
        async with websockets.connect(url, close_timeout=10) as ws:
            # Send session_start
            await ws.send(json.dumps({"type": "session_start"}))
            if verbose:
                logger.info(f"[{persona_id}] Connected, session started")

            # Wait for initial "listening" status
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=RESPONSE_TIMEOUT)
                    data = json.loads(msg)
                    if verbose:
                        logger.debug(f"[{persona_id}] ← {data.get('type', '?')}")
                    if data.get("type") == "status" and data.get("state") == "listening":
                        break
            except asyncio.TimeoutError:
                logger.warning(f"[{persona_id}] Timeout waiting for initial listening status")

            for turn in range(1, num_turns + 1):
                # Generate persona message
                user_msg = await generate_persona_message(
                    client, persona, conversation_history, turn, num_turns
                )
                if verbose:
                    logger.info(f"[{persona_id}] Turn {turn} → {user_msg}")

                # Send text message
                await ws.send(json.dumps({"type": "text", "text": user_msg}))
                conversation_history.append({"role": "user", "text": user_msg})

                # Wait for transcript response
                response_text = ""
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=RESPONSE_TIMEOUT)
                        data = json.loads(msg)
                        if data.get("type") == "transcript" and data.get("final"):
                            response_text = data.get("text", "")
                            break
                        elif data.get("type") == "status":
                            continue
                except asyncio.TimeoutError:
                    logger.warning(f"[{persona_id}] Turn {turn}: timeout waiting for response")
                    response_text = "[TIMEOUT - NO RESPONSE]"

                if verbose:
                    logger.info(f"[{persona_id}] Turn {turn} ← {response_text[:100]}...")

                conversation_history.append({"role": "assistant", "text": response_text})
                turns.append((user_msg, response_text))

                # Small delay between turns
                await asyncio.sleep(0.5)

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"[{persona_id}] WebSocket closed: {e}")
        meta["error"] = f"Connection closed: {e}"
    except Exception as e:
        logger.error(f"[{persona_id}] Error: {e}")
        meta["error"] = str(e)

    meta["duration_s"] = round(time.time() - t_start, 1)
    meta["turns_completed"] = len(turns)
    return turns, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(
    persona_ids: list[str],
    num_turns: int = 15,
    verbose: bool = False,
    use_judge: bool = False,
) -> tuple[list[ConversationScore], dict[str, dict]]:
    """Run conversations for all specified personas and score them.

    Returns (deterministic_scores, persona_results) where persona_results
    is keyed by persona_id with deterministic and optional llm_judge data.
    """
    scores: list[ConversationScore] = []
    persona_results: dict[str, dict] = {}
    total_judge_tokens = {"input": 0, "output": 0}

    for pid in persona_ids:
        persona = load_persona(pid)
        logger.info(f"Running persona: {persona['name']} ({pid})")

        turns, meta = await run_conversation(persona, num_turns, verbose)

        # Deterministic scoring
        scorer = ConvoScorer(user_name=persona["name"].split()[0])
        score = scorer.score_conversation(pid, turns)
        scores.append(score)

        result: dict = {
            "deterministic": {
                "level": score.level,
                "fails": score.fail_count,
                "warnings": score.warning_count,
                "turns": len(turns),
            },
        }

        logger.info(f"  → {score.level.upper()}: {score.fail_count} fails, {score.warning_count} warnings in {len(turns)} turns ({meta.get('duration_s', '?')}s)")

        # LLM Judge (optional)
        if use_judge and turns:
            logger.info(f"  → Running LLM judge for {pid}...")
            judge_result = judge_conversation(turns, persona)
            result["llm_judge"] = judge_result

            usage = judge_result.get("_usage", {})
            total_judge_tokens["input"] += usage.get("input_tokens", 0)
            total_judge_tokens["output"] += usage.get("output_tokens", 0)

            if judge_result.get("scores"):
                avg = sum(v for v in judge_result["scores"].values() if v > 0) / max(1, sum(1 for v in judge_result["scores"].values() if v > 0))
                logger.info(f"  → Judge avg: {avg:.1f}/5 | {judge_result.get('overall_assessment', '')[:80]}")

        persona_results[pid] = result

    if use_judge:
        logger.info(f"Judge totals: {total_judge_tokens['input']} input + {total_judge_tokens['output']} output tokens")

    return scores, persona_results


def main():
    parser = argparse.ArgumentParser(description="ConvoSim — EverNear conversation tester")
    parser.add_argument("--personas", default="critical", help="all | critical | comma-separated ids")
    parser.add_argument("--turns", type=int, default=15, help="Number of turns per conversation")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test users and exit")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring (costs API tokens)")
    parser.add_argument("--full", action="store_true", help="Alias for --personas all --judge")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.cleanup:
        cleanup_test_users()
        return

    # Handle --full alias
    if args.full:
        args.personas = "all"
        args.judge = True

    # Resolve persona list
    if args.personas == "all":
        persona_ids = list_personas()
    elif args.personas == "critical":
        persona_ids = CRITICAL_PERSONAS
    else:
        persona_ids = [p.strip() for p in args.personas.split(",")]

    logger.info(f"ConvoSim starting: {len(persona_ids)} personas, {args.turns} turns each" +
                (" [+judge]" if args.judge else ""))

    scores, persona_results = asyncio.run(run_all(persona_ids, args.turns, args.verbose, args.judge))

    # Save trend data if judge was used
    trend_comparison = None
    if args.judge:
        from .trends import save_run, get_previous_run, compare_runs
        previous = get_previous_run()
        trend_path = save_run(persona_results)
        logger.info(f"Trend data saved to {trend_path}")

        if previous:
            from .trends import get_latest_run
            current = get_latest_run()
            if current:
                trend_comparison = compare_runs(current, previous)
                if trend_comparison.get("warnings"):
                    for w in trend_comparison["warnings"]:
                        logger.warning(f"TREND: {w}")

    # Report
    meta = {
        "personas": len(persona_ids),
        "turns_per_persona": args.turns,
        "ws_url": WS_URL,
        "judge_enabled": args.judge,
    }
    report_path = print_and_save(scores, meta, persona_results if args.judge else None, trend_comparison)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
