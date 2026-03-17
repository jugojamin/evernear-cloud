"""Response Validation Layer — rule-based, deterministic, no LLM calls.

Sits between LLM output and TTS input in the pipeline.
Every response passes through. No bypass.
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from server.memory_cache import MemoryCache

logger = logging.getLogger(__name__)


# --- Result Types ---

class ValidationAction(str, Enum):
    ALLOW = "ALLOW"
    REWRITE = "REWRITE"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"

# Severity ordering for priority resolution
_ACTION_SEVERITY = {
    ValidationAction.ALLOW: 0,
    ValidationAction.REWRITE: 1,
    ValidationAction.BLOCK: 2,
    ValidationAction.ESCALATE: 3,
}


@dataclass
class ValidationResult:
    action: ValidationAction
    response: str
    original_response: str
    user_input: str = ""
    checks_triggered: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# --- Pattern Lists ---
# All seed lists are V1 — intentionally small and literal.
# Expansion is evidence-based, from intervention log review only.

# Check 1: Tier A Escalation — scans user_input
TIER_A_TRIGGERS: dict[str, list[str]] = {
    "fall": [
        "i fell",
        "i just fell",
        "i've fallen",
        "i'm on the floor",
        "i can't get up",
        "i just can't get up",
        "i fell down",
    ],
    "medical_emergency": [
        "i can't breathe",
        "i just can't breathe",
        "i really can't breathe",
        "my chest hurts",
        "chest pain",
        "i think i'm having a heart attack",
        "i think i'm having a stroke",
        "i can't feel my arm",
        "i can't feel my leg",
    ],
    "self_harm": [
        "i want to die",
        "i just want to die",
        "i really want to die",
        "i don't want to be here anymore",
        "i just don't want to be here anymore",
        "i really don't want to be here anymore",
        "i want to end it",
        "i just want to end it",
        "i really want to end it",
        "i'm going to hurt myself",
        "what's the point of living",
    ],
}
# Last reviewed: 2026-03-15 (filler-word variants added — live validation miss)

TIER_A_EXCLUSIONS: dict[str, list[str]] = {
    "fall": ["fell asleep", "fell in love", "fell behind", "fell apart", "fell for"],
    "medical_emergency": [],
    "self_harm": [],
}
# Last reviewed: 2026-03-15 (initial V1 — North)

TIER_A_RESPONSES: dict[str, str] = {
    "fall": "I hear you. Stay still and try to stay calm. I'm letting {caregiver_name} know right now. I'll stay right here with you.",
    "medical_emergency": "That sounds serious, and I want to make sure you're safe. I'm reaching out to {caregiver_name} right now. Stay with me.",
    "self_harm": "I'm really glad you told me that. You matter, and I want to make sure someone who cares about you knows how you're feeling. I'm reaching out to {caregiver_name} now. I'm here with you.",
}

# Priority order for multiple Tier A matches
TIER_A_PRIORITY = ["medical_emergency", "self_harm", "fall"]

# Check 2: Medical Advice — scans response
MEDICAL_HARD_BLOCKS: list[str] = [
    "you should take",
    "try taking",
    "stop taking",
    "increase your dose",
    "decrease your dose",
    "your symptoms suggest",
    "you probably have",
    "you might have",
]
# Last reviewed: 2026-03-15 (initial V1 seed list — North)

MEDICAL_SOFT_FLAGS: list[str] = [
    "that sounds like",
]

ADVISORY_LANGUAGE: list[str] = ["you should", "i think you", "try ", "it might help"]

COMMON_MEDICATIONS: list[str] = [
    "lisinopril", "amlodipine", "metoprolol", "losartan",
    "atorvastatin", "simvastatin", "rosuvastatin",
    "metformin", "glipizide", "insulin",
    "warfarin", "eliquis", "xarelto",
    "omeprazole", "pantoprazole",
    "acetaminophen", "ibuprofen", "gabapentin",
    "levothyroxine",
    "aricept", "donepezil",
]
# Last reviewed: 2026-03-15 (initial V1 starter list — North)

MEDICAL_CONDITIONS: list[str] = [
    "diabetes", "hypertension", "arthritis", "dementia", "alzheimer",
    "stroke", "heart disease", "heart failure", "pneumonia", "infection",
    "anemia", "depression", "anxiety", "fracture", "blood pressure",
    "blood sugar", "cholesterol",
]
# Last reviewed: 2026-03-15 (initial V1 starter list — North)

# Check 3: Memory Certainty — scans response
PERSONAL_ASSERTION_PATTERNS: list[str] = [
    "your ",  # "Your daughter", "Your husband" etc. — refined in _check_memory_certainty
    "you told me about",
    "you told me that",
    "you used to",
    "remember when you",
    "you mentioned",
]
# Note: "you told me" without "about/that" excluded — matches "glad you told me" (false positive)

FAMILY_PERSON_WORDS: list[str] = [
    "daughter", "son", "husband", "wife", "sister", "brother",
    "mother", "father", "mom", "dad", "grandchild", "grandson",
    "granddaughter", "friend", "neighbor", "caregiver",
    "nephew", "niece", "uncle", "aunt", "cousin",
]
# "doctor" and "nurse" removed — role references, not personal-memory entities.
# EverNear uses "your doctor" in medical redirects, causing Check 3 false positives.

# Check 4: Unsafe Reassurance — scans response + user_input
DISMISSAL_PATTERNS: list[str] = [
    "you don't need to worry",
    "i'm sure it's nothing",
    "that's completely normal",
    "you're probably fine",
    "don't think about it",
    "it's not a big deal",
    "there's nothing to worry about",
    "everyone feels that way",
]
# Last reviewed: 2026-03-15 (initial V1 seed list — North)

CONTEXT_AMPLIFIERS: list[str] = [
    "pain", "chest", "breathing", "dizzy",
    "fall", "fell", "scared", "alone",
    "can't remember", "confused", "lost", "help",
]

# Check 5: False Certainty — scans response
EXTERNAL_FACT_CATEGORIES: list[str] = [
    "weather", "temperature", "degrees", "forecast",
    "news", "president", "election",
    "score",
    "stock", "market",
]
# Last reviewed: 2026-03-15 (initial V1 seed list — North)

HEDGE_WORDS: list[str] = ["i think", "i'm not sure", "if i remember", "i believe", "maybe"]


# --- Replacement Responses (all static strings) ---

MEDICAL_BLOCK_RESPONSE = "That's something worth asking your doctor about. I want to make sure you get the right answer on that."
MEDICAL_REWRITE_RESPONSE = "That sounds like something worth mentioning to your doctor. They'd know best."
MEMORY_REWRITE_TEMPLATE = "I want to make sure I'm remembering that right — was it {detail}?"
MEMORY_BLOCK_RESPONSE = "I don't think you've told me about that yet. I'd love to hear."
REASSURANCE_BLOCK_RESPONSE = "That sounds uncomfortable, and I'm glad you told me. That's something your doctor should know about. Would you like me to remind you to mention it to them?"
REASSURANCE_REWRITE_RESPONSE = "I hear you. If it keeps bothering you, it might be worth mentioning next time you see your doctor."
FALSE_CERTAINTY_RESPONSE = "I wish I could check that for you, but I don't have a way to look that up. I don't want to guess and get it wrong."
UNIVERSAL_FALLBACK = "I'm sorry, I lost my thought for a moment. Could you say that again?"


@dataclass
class SessionContext:
    """Minimal session context needed by the validator."""
    caregiver_name: str = ""
    user_id: str = ""


class ResponseValidator:
    """Rule-based response validation. Deterministic. No LLM calls.

    Sits between LLM output and TTS input in the pipeline.
    Every response passes through. No bypass.
    """

    UNIVERSAL_FALLBACK = UNIVERSAL_FALLBACK

    def validate(
        self,
        response: str,
        user_input: str,
        memory_cache: MemoryCache | None,
        session_context: SessionContext | None = None,
    ) -> ValidationResult:
        """Run all checks. Return highest-severity result.

        Returns ValidationResult with action, response, checks_triggered, latency_ms.
        """
        start = time.monotonic()
        ctx = session_context or SessionContext()
        original_response = response

        results: list[tuple[ValidationAction, str, str, dict]] = []

        # Check 1: Tier A Escalation (scans user_input)
        tier_a = self._check_tier_a(user_input, ctx)
        if tier_a:
            results.append(tier_a)

        # Check 2: Medical Advice (scans response)
        medical = self._check_medical_advice(response)
        if medical:
            results.append(medical)

        # Check 3: Memory Certainty (scans response + memory_cache)
        memory = self._check_memory_certainty(response, memory_cache)
        if memory:
            results.append(memory)

        # Check 4: Unsafe Reassurance (scans response + user_input)
        reassurance = self._check_unsafe_reassurance(response, user_input)
        if reassurance:
            results.append(reassurance)

        # Check 5: False Certainty (scans response)
        certainty = self._check_false_certainty(response)
        if certainty:
            results.append(certainty)

        latency = (time.monotonic() - start) * 1000

        # Resolve: highest severity wins
        if not results:
            return ValidationResult(
                action=ValidationAction.ALLOW,
                response=response,
                original_response=original_response,
                user_input=user_input,
                latency_ms=latency,
            )

        # Sort by severity descending
        results.sort(key=lambda r: _ACTION_SEVERITY[r[0]], reverse=True)
        best_action, best_response, best_check, best_details = results[0]
        all_checks = [r[2] for r in results]

        return ValidationResult(
            action=best_action,
            response=best_response,
            original_response=original_response,
            user_input=user_input,
            checks_triggered=all_checks,
            latency_ms=latency,
            details=best_details,
        )

    def _check_tier_a(
        self, user_input: str, ctx: SessionContext
    ) -> tuple[ValidationAction, str, str, dict] | None:
        """Check 1: Tier A Escalation — life safety triggers in user input."""
        input_lower = user_input.lower()
        triggered_categories: list[str] = []

        for category, patterns in TIER_A_TRIGGERS.items():
            for pattern in patterns:
                if pattern in input_lower:
                    # Check exclusions
                    exclusions = TIER_A_EXCLUSIONS.get(category, [])
                    excluded = any(exc in input_lower for exc in exclusions)
                    if not excluded:
                        triggered_categories.append(category)
                        break  # One match per category is enough

        if not triggered_categories:
            return None

        # Pick highest priority
        primary = None
        for cat in TIER_A_PRIORITY:
            if cat in triggered_categories:
                primary = cat
                break
        if primary is None:
            primary = triggered_categories[0]

        caregiver_name = ctx.caregiver_name or "someone who can help"
        response_text = TIER_A_RESPONSES[primary].format(caregiver_name=caregiver_name)

        return (
            ValidationAction.ESCALATE,
            response_text,
            f"tier_a:{primary}",
            {"all_categories": triggered_categories, "primary": primary},
        )

    def _check_medical_advice(
        self, response: str
    ) -> tuple[ValidationAction, str, str, dict] | None:
        """Check 2: Medical Advice — block or rewrite medical guidance."""
        response_lower = response.lower()

        # Hard blockers
        for pattern in MEDICAL_HARD_BLOCKS:
            if pattern in response_lower:
                return (
                    ValidationAction.BLOCK,
                    MEDICAL_BLOCK_RESPONSE,
                    "medical:hard_block",
                    {"pattern": pattern},
                )

        # Soft flags: "that sounds like" + medical condition within 5 words
        for flag in MEDICAL_SOFT_FLAGS:
            idx = response_lower.find(flag)
            if idx >= 0:
                # Check for medical condition within next ~40 chars (roughly 5 words)
                window = response_lower[idx + len(flag):idx + len(flag) + 40]
                for condition in MEDICAL_CONDITIONS:
                    if condition in window:
                        # Check for advisory language anywhere in response
                        has_advisory = any(adv in response_lower for adv in ADVISORY_LANGUAGE)
                        if has_advisory:
                            return (
                                ValidationAction.REWRITE,
                                MEDICAL_REWRITE_RESPONSE,
                                "medical:soft_flag",
                                {"flag": flag, "condition": condition},
                            )

        # Medication name + advisory language (not in reminder context)
        for med in COMMON_MEDICATIONS:
            if med in response_lower:
                # Check if it's a reminder context (allowed)
                reminder_phrases = ["time for your", "reminder about your", "don't forget your"]
                is_reminder = any(rp in response_lower for rp in reminder_phrases)
                if not is_reminder:
                    has_advisory = any(adv in response_lower for adv in ADVISORY_LANGUAGE)
                    if has_advisory:
                        return (
                            ValidationAction.REWRITE,
                            MEDICAL_REWRITE_RESPONSE,
                            "medical:medication_advisory",
                            {"medication": med},
                        )

        return None

    def _check_memory_certainty(
        self, response: str, memory_cache: MemoryCache | None
    ) -> tuple[ValidationAction, str, str, dict] | None:
        """Check 3: Memory Certainty — validate personal assertions against stored memories."""
        response_lower = response.lower()

        # Detect personal assertion patterns
        has_assertion = False
        assertion_detail = ""

        # "Your [family/person word]"
        for word in FAMILY_PERSON_WORDS:
            pattern = f"your {word}"
            if pattern in response_lower:
                has_assertion = True
                assertion_detail = pattern
                break

        # Non-assertion phrases that contain assertion patterns (false positives)
        assertion_exclusions = [
            "glad you told me",
            "thank you for telling me",
            "happy you told me",
            "glad you mentioned",
            "thank you for mentioning",
        ]

        if not has_assertion:
            for pattern in PERSONAL_ASSERTION_PATTERNS[1:]:  # Skip "your " — handled above
                if pattern in response_lower:
                    # Check exclusions
                    excluded = any(exc in response_lower for exc in assertion_exclusions)
                    if not excluded:
                        has_assertion = True
                        assertion_detail = pattern
                        break

        if not has_assertion:
            return None

        # If no memory cache available, block personal assertions by default
        if memory_cache is None:
            return (
                ValidationAction.BLOCK,
                MEMORY_BLOCK_RESPONSE,
                "memory:cache_unavailable",
                {"assertion": assertion_detail},
            )

        # Determine category from assertion
        category = "family"  # Default for person-word assertions
        for cat_keyword, cat_name in [
            ("health", "health"), ("doctor", "health"), ("medication", "health"),
            ("routine", "routine"), ("church", "faith"), ("prayer", "faith"),
        ]:
            if cat_keyword in response_lower:
                category = cat_name
                break

        # Extract entities from the assertion (simple word extraction)
        # Get proper nouns — words that start with uppercase in original response
        words = response.split()
        entities = set()
        for w in words:
            cleaned = w.strip(".,!?;:'\"")
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                entities.add(cleaned)

        memory, has_overlap = memory_cache.find_by_category_and_entities(category, entities)

        if memory and has_overlap:
            # Check confidence
            confidence = memory.get("confidence", 0.5)
            verified = memory.get("verified", False)
            if confidence >= 0.7 or verified:
                return None  # ALLOW
            else:
                # Low confidence — rewrite to verify
                detail = memory.get("content", "that")
                return (
                    ValidationAction.REWRITE,
                    MEMORY_REWRITE_TEMPLATE.format(detail=detail),
                    "memory:low_confidence",
                    {"memory_content": detail, "confidence": confidence},
                )
        elif memory and not has_overlap:
            # Same category but no entity overlap
            return (
                ValidationAction.BLOCK,
                MEMORY_BLOCK_RESPONSE,
                "memory:no_entity_overlap",
                {"assertion": assertion_detail, "category": category},
            )
        else:
            # No memory in category
            return (
                ValidationAction.BLOCK,
                MEMORY_BLOCK_RESPONSE,
                "memory:no_memory",
                {"assertion": assertion_detail, "category": category},
            )

    def _check_unsafe_reassurance(
        self, response: str, user_input: str
    ) -> tuple[ValidationAction, str, str, dict] | None:
        """Check 4: Unsafe Reassurance — catch dismissive responses to serious concerns."""
        response_lower = response.lower()
        input_lower = user_input.lower()

        triggered_dismissal = None
        for pattern in DISMISSAL_PATTERNS:
            if pattern in response_lower:
                triggered_dismissal = pattern
                break

        if not triggered_dismissal:
            return None

        # Check for context amplifiers in user input
        has_amplifier = any(amp in input_lower for amp in CONTEXT_AMPLIFIERS)

        if has_amplifier:
            return (
                ValidationAction.BLOCK,
                REASSURANCE_BLOCK_RESPONSE,
                "reassurance:dismissal_with_amplifier",
                {"dismissal": triggered_dismissal},
            )
        else:
            return (
                ValidationAction.REWRITE,
                REASSURANCE_REWRITE_RESPONSE,
                "reassurance:dismissal_only",
                {"dismissal": triggered_dismissal},
            )

    def _check_false_certainty(
        self, response: str
    ) -> tuple[ValidationAction, str, str, dict] | None:
        """Check 5: False Certainty — catch claims about external facts EN can't know."""
        response_lower = response.lower()

        has_external_fact = False
        triggered_keyword = ""
        for keyword in EXTERNAL_FACT_CATEGORIES:
            if keyword in response_lower:
                has_external_fact = True
                triggered_keyword = keyword
                break

        if not has_external_fact:
            return None

        # Check for hedge words
        has_hedge = any(hw in response_lower for hw in HEDGE_WORDS)

        if has_hedge:
            # Hedged claims are allowed but logged
            return None  # ALLOW — logging happens at the pipeline level

        return (
            ValidationAction.REWRITE,
            FALSE_CERTAINTY_RESPONSE,
            "false_certainty:unhedged",
            {"keyword": triggered_keyword},
        )
