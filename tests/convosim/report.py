"""ConvoSim report generator."""

from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from .scorer import ConversationScore, Level

REPORTS_DIR = Path(__file__).parent / "reports"


def _level_icon(level: Level) -> str:
    return {"pass": "✅", "warning": "⚠️", "fail": "❌"}[level]


def generate_report(scores: list[ConversationScore], meta: dict | None = None) -> str:
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


def print_and_save(scores: list[ConversationScore], meta: dict | None = None) -> Path:
    """Generate, print, and save report."""
    text = generate_report(scores, meta)
    print(text)
    return save_report(text)
