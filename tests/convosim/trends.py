"""ConvoSim trends — store and compare run scores over time."""

from __future__ import annotations
import json, logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("convosim.trends")

HISTORY_DIR = Path(__file__).parent / "history"
DIMENSIONS = ["warmth", "pacing", "respect", "listening", "boundaries", "safety", "naturalness", "memory"]


def save_run(
    persona_results: dict[str, dict[str, Any]],
    timestamp: str | None = None,
) -> Path:
    """Save a run to history. persona_results = {persona_id: {"deterministic": {...}, "llm_judge": {...}}}"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y-%m-%d-%H%M%S")

    # Compute averages across personas that have judge scores
    averages: dict[str, float] = {}
    counts: dict[str, int] = {d: 0 for d in DIMENSIONS}
    totals: dict[str, float] = {d: 0.0 for d in DIMENSIONS}

    for pid, data in persona_results.items():
        judge = data.get("llm_judge", {})
        scores = judge.get("scores", {})
        for dim in DIMENSIONS:
            val = scores.get(dim, 0)
            if val > 0:
                totals[dim] += val
                counts[dim] += 1

    for dim in DIMENSIONS:
        if counts[dim] > 0:
            averages[dim] = round(totals[dim] / counts[dim], 2)

    run_data = {
        "timestamp": ts,
        "personas": persona_results,
        "averages": averages,
    }

    path = HISTORY_DIR / f"{ts}.json"
    path.write_text(json.dumps(run_data, indent=2))
    logger.info(f"Saved trend data to {path}")
    return path


def get_previous_run() -> dict | None:
    """Load the most recent previous run from history."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(HISTORY_DIR.glob("*.json"))
    if not files:
        return None
    # Return the second-to-last if we just saved, or the last
    # Since save happens before compare, take the second-to-last
    if len(files) >= 2:
        return json.loads(files[-2].read_text())
    return None


def get_latest_run() -> dict | None:
    """Load the most recent run."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(HISTORY_DIR.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def compare_runs(current: dict, previous: dict) -> dict[str, Any]:
    """Compare two runs and return deltas.

    Returns: {"deltas": {"warmth": +0.3, ...}, "warnings": ["safety dropped by 0.7 ⚠️"], "previous_timestamp": "..."}
    """
    curr_avg = current.get("averages", {})
    prev_avg = previous.get("averages", {})

    deltas: dict[str, float] = {}
    warnings: list[str] = []

    for dim in DIMENSIONS:
        c = curr_avg.get(dim, 0)
        p = prev_avg.get(dim, 0)
        if p > 0 and c > 0:
            delta = round(c - p, 2)
            deltas[dim] = delta
            if delta < -0.5:
                warnings.append(f"{dim} dropped by {abs(delta):.1f} ⚠️")
        elif c > 0:
            deltas[dim] = 0  # no previous to compare

    return {
        "deltas": deltas,
        "warnings": warnings,
        "previous_timestamp": previous.get("timestamp", "unknown"),
    }
