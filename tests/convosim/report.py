"""ConvoSim report generator."""

from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from .scorer import ConversationScore, Level

REPORTS_DIR = Path(__file__).parent / "reports"

DIMENSIONS = ["warmth", "pacing", "respect", "listening", "boundaries", "safety", "naturalness", "memory"]


def _level_icon(level: Level) -> str:
    return {"pass": "✅", "warning": "⚠️", "fail": "❌"}[level]


def _score_bar(score: int | float) -> str:
    """Visual bar for 1-5 score."""
    s = int(round(score))
    return "█" * s + "░" * (5 - s) + f" {score}"


def generate_report(
    scores: list[ConversationScore],
    meta: dict | None = None,
    judge_results: dict[str, dict] | None = None,
    trend_comparison: dict | None = None,
) -> str:
    """Generate a text report from scored conversations."""
    lines: list[str] = []
    now = datetime.now()
    lines.append("=" * 70)
    lines.append(f"CONVOSIM REPORT — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    if meta:
        for k, v in meta.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Summary
    total_pass = sum(1 for s in scores if s.level == "pass")
    total_warn = sum(1 for s in scores if s.level == "warning")
    total_fail = sum(1 for s in scores if s.level == "fail")
    lines.append(f"SUMMARY: {len(scores)} personas — ✅ {total_pass} pass, ⚠️ {total_warn} warning, ❌ {total_fail} fail")
    lines.append("-" * 70)

    for cs in scores:
        lines.append(f"\n{_level_icon(cs.level)} {cs.persona_id} — {len(cs.turns)} turns, {cs.fail_count} fails, {cs.warning_count} warnings")

        for ts in cs.turns:
            if ts.issues:
                for issue in ts.issues:
                    lines.append(f"    Turn {ts.turn} [{issue.level.upper()}] {issue.code}: {issue.detail}")
                    if issue.level == "fail":
                        lines.append(f"      User: {ts.user_text[:100]}")
                        lines.append(f"      Response: {ts.response_text[:150]}")

        # LLM Judge results for this persona
        if judge_results and cs.persona_id in judge_results:
            jr = judge_results[cs.persona_id].get("llm_judge", {})
            judge_scores = jr.get("scores", {})
            if judge_scores and any(v > 0 for v in judge_scores.values()):
                lines.append(f"    ┌─ LLM Judge Scores:")
                for dim in DIMENSIONS:
                    val = judge_scores.get(dim, 0)
                    if val > 0:
                        marker = " ⚠️" if val < 3 else ""
                        lines.append(f"    │  {dim:12s} {_score_bar(val)}{marker}")

                # Flagged turns
                for flag in jr.get("flagged_turns", []):
                    lines.append(f"    │  ⚑ Turn {flag.get('turn', '?')} [{flag.get('dimension', '?')}]: {flag.get('issue', '')}")

                # Negative experience flags
                for nef in jr.get("negative_experience_flags", []):
                    lines.append(f"    │  😟 Turn {nef.get('turn', '?')} ({nef.get('reaction', '?')}): {nef.get('detail', '')}")

                assessment = jr.get("overall_assessment", "")
                if assessment:
                    lines.append(f"    │  Assessment: {assessment[:200]}")
                lines.append(f"    └─")

    # --- LLM Judge Averages ---
    if judge_results:
        lines.append("\n" + "=" * 70)
        lines.append("LLM JUDGE — OVERALL AVERAGES")
        lines.append("-" * 70)

        totals: dict[str, float] = {d: 0.0 for d in DIMENSIONS}
        counts: dict[str, int] = {d: 0 for d in DIMENSIONS}

        for pid, data in judge_results.items():
            jr = data.get("llm_judge", {})
            for dim in DIMENSIONS:
                val = jr.get("scores", {}).get(dim, 0)
                if val > 0:
                    totals[dim] += val
                    counts[dim] += 1

        for dim in DIMENSIONS:
            if counts[dim] > 0:
                avg = round(totals[dim] / counts[dim], 2)
                marker = " ⚠️" if avg < 3 else ""
                lines.append(f"  {dim:12s} {_score_bar(avg)}{marker}")

    # --- Trend Comparison ---
    if trend_comparison:
        lines.append("\n" + "=" * 70)
        lines.append(f"TREND COMPARISON (vs {trend_comparison.get('previous_timestamp', 'unknown')})")
        lines.append("-" * 70)

        deltas = trend_comparison.get("deltas", {})
        for dim in DIMENSIONS:
            delta = deltas.get(dim)
            if delta is not None:
                sign = "+" if delta >= 0 else ""
                warn = " ⚠️" if delta < -0.5 else ""
                lines.append(f"  {dim:12s} {sign}{delta:.2f}{warn}")

        warnings = trend_comparison.get("warnings", [])
        if warnings:
            lines.append("")
            for w in warnings:
                lines.append(f"  🚨 {w}")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


def save_report(report_text: str) -> Path:
    """Save report to reports/ directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y-%m-%d-%H%M%S") + ".txt"
    path = REPORTS_DIR / filename
    path.write_text(report_text)
    return path


def print_and_save(
    scores: list[ConversationScore],
    meta: dict | None = None,
    judge_results: dict[str, dict] | None = None,
    trend_comparison: dict | None = None,
) -> Path:
    """Generate, print, and save report."""
    text = generate_report(scores, meta, judge_results, trend_comparison)
    print(text)
    return save_report(text)
