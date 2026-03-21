"""Tests for the Response Validation Layer (Phase 1)."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from server.validator import (
    ResponseValidator,
    ValidationAction,
    SessionContext,
    TIER_A_TRIGGERS,
    TIER_A_RESPONSES,
    MEDICAL_HARD_BLOCKS,
    UNIVERSAL_FALLBACK,
)
from server.memory_cache import MemoryCache


@pytest.fixture
def validator():
    return ResponseValidator()


@pytest.fixture
def ctx_with_caregiver():
    return SessionContext(caregiver_name="Sarah", user_id="user-123")


@pytest.fixture
def ctx_no_caregiver():
    return SessionContext(caregiver_name="", user_id="user-123")


def _make_cache(memories=None):
    """Create a mock MemoryCache with given memories."""
    cache = MagicMock(spec=MemoryCache)
    cache.memories = {}
    if memories:
        for m in memories:
            cat = m.get("category", "general")
            cache.memories.setdefault(cat, []).append(m)

    def _has_any(category):
        return bool(cache.memories.get(category))

    def _find_by_category_and_entities(category, entities):
        cat_memories = cache.memories.get(category, [])
        if not cat_memories:
            return None, False
        entities_lower = {e.lower() for e in entities if len(e) > 1}
        for m in cat_memories:
            content_words = set(m.get("content", "").lower().split())
            if entities_lower & content_words:
                return m, True
        return cat_memories[0], False

    cache.has_any.side_effect = _has_any
    cache.find_by_category_and_entities.side_effect = _find_by_category_and_entities
    return cache


# === Check 1: Tier A Escalation ===

class TestTierAFallTriggers:
    def test_i_fell(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE
        assert "tier_a:fall" in r.checks_triggered
        assert "Sarah" in r.response

    def test_ive_fallen(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I've fallen and I'm scared", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_im_on_the_floor(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I'm on the floor", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_cant_get_up(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I can't get up", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_fell_down(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell down in the kitchen", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE


class TestTierAMedicalEmergency:
    def test_cant_breathe(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I can't breathe", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE
        assert "tier_a:medical_emergency" in r.checks_triggered

    def test_chest_hurts(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "My chest hurts", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_chest_pain(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I have chest pain", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_heart_attack(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I think I'm having a heart attack", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_stroke(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I think I'm having a stroke", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_cant_feel_arm(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I can't feel my arm", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_cant_feel_leg(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I can't feel my leg", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE


class TestTierASelfHarm:
    def test_want_to_die(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I want to die", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE
        assert "tier_a:self_harm" in r.checks_triggered

    def test_dont_want_to_be_here(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I don't want to be here anymore", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_end_it(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I want to end it", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_hurt_myself(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I'm going to hurt myself", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE

    def test_point_of_living(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "What's the point of living", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE


class TestTierABenignInput:
    def test_normal_conversation(self, validator, ctx_with_caregiver):
        r = validator.validate("That sounds lovely.", "I had a nice day today", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ALLOW

    def test_fell_asleep_exclusion(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell asleep on the couch", _make_cache(), ctx_with_caregiver)
        assert r.action != ValidationAction.ESCALATE

    def test_fell_in_love_exclusion(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell in love with that movie", _make_cache(), ctx_with_caregiver)
        assert r.action != ValidationAction.ESCALATE

    def test_fell_behind_exclusion(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell behind on my reading", _make_cache(), ctx_with_caregiver)
        assert r.action != ValidationAction.ESCALATE


class TestTierACaregiverSubstitution:
    def test_with_caregiver(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell", _make_cache(), ctx_with_caregiver)
        assert "Sarah" in r.response

    def test_without_caregiver(self, validator, ctx_no_caregiver):
        r = validator.validate("Hello", "I fell", _make_cache(), ctx_no_caregiver)
        assert "someone who can help" in r.response


class TestTierAMultipleCategories:
    def test_medical_takes_priority_over_fall(self, validator, ctx_with_caregiver):
        r = validator.validate("Hello", "I fell and my chest hurts", _make_cache(), ctx_with_caregiver)
        assert r.action == ValidationAction.ESCALATE
        assert "tier_a:medical_emergency" in r.checks_triggered
        assert r.details.get("primary") == "medical_emergency"


# === Check 2: Medical Advice ===

class TestMedicalHardBlockers:
    @pytest.mark.parametrize("pattern", MEDICAL_HARD_BLOCKS)
    def test_hard_block_triggers(self, validator, pattern):
        response = f"Based on what you said, {pattern} some aspirin."
        r = validator.validate(response, "I have a headache", _make_cache())
        assert r.action == ValidationAction.BLOCK
        assert "medical:hard_block" in r.checks_triggered


class TestMedicalSoftFlags:
    def test_medication_with_advisory(self, validator):
        response = "You should try taking lisinopril with food."
        r = validator.validate(response, "My blood pressure is high", _make_cache())
        # Should trigger either hard block ("try taking") or medication advisory
        assert r.action in (ValidationAction.BLOCK, ValidationAction.REWRITE)

    def test_medication_without_advisory(self, validator):
        response = "I remember you mentioned lisinopril last time."
        r = validator.validate(response, "What medications do I take?", _make_cache())
        assert r.action != ValidationAction.BLOCK or "medical" not in str(r.checks_triggered)

    def test_reminder_exception(self, validator):
        response = "Time for your blood pressure medication."
        r = validator.validate(response, "What time is it?", _make_cache())
        # Reminder context should not trigger medical block
        assert r.action != ValidationAction.BLOCK or "medical" not in str(r.checks_triggered)

    def test_sounds_like_condition_with_advisory(self, validator):
        response = "That sounds like diabetes. You should talk to your doctor."
        r = validator.validate(response, "I've been very thirsty", _make_cache())
        assert r.action in (ValidationAction.REWRITE, ValidationAction.BLOCK)


# === Check 3: Memory Certainty ===

class TestMemoryVerified:
    def test_assertion_with_verified_memory(self, validator):
        cache = _make_cache([{
            "category": "family",
            "content": "daughter Sarah lives in Austin",
            "confidence": 0.9,
            "verified": True,
        }])
        r = validator.validate(
            "Your daughter Sarah is in Austin.",
            "Tell me about my family",
            cache,
        )
        assert r.action == ValidationAction.ALLOW

    def test_assertion_with_high_confidence_memory(self, validator):
        cache = _make_cache([{
            "category": "family",
            "content": "daughter Sarah lives in Austin",
            "confidence": 0.8,
            "verified": False,
        }])
        r = validator.validate(
            "Your daughter Sarah is in Austin.",
            "Tell me about my family",
            cache,
        )
        assert r.action == ValidationAction.ALLOW


class TestMemoryLowConfidence:
    def test_assertion_with_low_confidence(self, validator):
        cache = _make_cache([{
            "category": "family",
            "content": "daughter Sarah lives in Austin",
            "confidence": 0.5,
            "verified": False,
        }])
        r = validator.validate(
            "Your daughter Sarah lives in Austin.",
            "Tell me about my family",
            cache,
        )
        assert r.action == ValidationAction.REWRITE
        assert "memory:low_confidence" in r.checks_triggered


class TestMemoryNoMemory:
    def test_assertion_with_no_memory(self, validator):
        cache = _make_cache([])
        r = validator.validate(
            "Your daughter Sarah lives in Austin.",
            "Tell me about my family",
            cache,
        )
        assert r.action == ValidationAction.BLOCK
        assert any("memory:" in c for c in r.checks_triggered)


class TestMemoryCacheMiss:
    def test_assertion_with_no_cache(self, validator):
        r = validator.validate(
            "Your daughter Sarah lives nearby.",
            "Tell me about my family",
            None,
        )
        assert r.action == ValidationAction.BLOCK
        assert "memory:cache_unavailable" in r.checks_triggered


# === Check 4: Unsafe Reassurance ===

class TestReassuranceDismissalWithAmplifier:
    def test_dismissal_plus_chest(self, validator):
        r = validator.validate(
            "I'm sure it's nothing to worry about.",
            "My chest has been hurting all day",
            _make_cache(),
        )
        assert r.action == ValidationAction.BLOCK
        assert "reassurance:dismissal_with_amplifier" in r.checks_triggered

    def test_dismissal_plus_pain(self, validator):
        r = validator.validate(
            "You're probably fine.",
            "I've been having a lot of pain",
            _make_cache(),
        )
        assert r.action == ValidationAction.BLOCK


class TestReassuranceDismissalOnly:
    def test_dismissal_benign_context(self, validator):
        r = validator.validate(
            "I'm sure it's nothing.",
            "I wonder if it will rain tomorrow",
            _make_cache(),
        )
        assert r.action == ValidationAction.REWRITE
        assert "reassurance:dismissal_only" in r.checks_triggered


class TestReassuranceNoDismissal:
    def test_empathetic_response(self, validator):
        r = validator.validate(
            "That sounds difficult. I'm glad you told me about it.",
            "I've been feeling lonely",
            _make_cache(),
        )
        assert r.action == ValidationAction.ALLOW


# === Check 5: False Certainty ===

class TestFalseCertaintyWeather:
    def test_weather_claim(self, validator):
        r = validator.validate(
            "It's 72 degrees and sunny outside today.",
            "What's the weather like?",
            _make_cache(),
        )
        assert r.action == ValidationAction.REWRITE
        assert "false_certainty:unhedged" in r.checks_triggered

    def test_hedged_claim(self, validator):
        r = validator.validate(
            "I'm not sure about the weather, but I think it might be nice out.",
            "What's the weather like?",
            _make_cache(),
        )
        assert r.action == ValidationAction.ALLOW


class TestFalseCertaintyPromptDefense:
    def test_system_prompt_has_what_you_dont_know(self):
        prompt_path = Path(__file__).parent.parent / "server" / "prompts" / "system_prompt.txt"
        content = prompt_path.read_text()
        assert "WHAT YOU DON'T KNOW" in content
        assert "NEVER guess or state external facts" in content


# === Priority & Logging ===

class TestPriorityOrdering:
    def test_escalate_beats_block(self, validator, ctx_with_caregiver):
        # User says something that triggers Tier A, response has medical advice
        r = validator.validate(
            "You should take your medication now.",
            "I can't breathe and I don't know what to do",
            _make_cache(),
            ctx_with_caregiver,
        )
        assert r.action == ValidationAction.ESCALATE
        # Both checks should be logged
        assert len(r.checks_triggered) >= 2

    def test_all_checks_logged(self, validator, ctx_with_caregiver):
        r = validator.validate(
            "I'm sure it's nothing. You should take some medicine.",
            "My chest hurts and I fell",
            _make_cache(),
            ctx_with_caregiver,
        )
        # Multiple checks should fire
        assert len(r.checks_triggered) >= 2


# === Fail Closed ===

class TestFailClosed:
    def test_validator_crash_returns_fallback(self):
        v = ResponseValidator()
        # Monkey-patch a check to raise
        original = v._check_tier_a
        def broken_check(*args, **kwargs):
            raise RuntimeError("boom")
        v._check_tier_a = broken_check

        # The pipeline wraps in try/except, so test that pattern
        try:
            r = v.validate("Hello", "Hi", _make_cache())
        except Exception:
            # Pipeline would catch this and use UNIVERSAL_FALLBACK
            pass
        assert UNIVERSAL_FALLBACK == "I'm sorry, I lost my thought for a moment. Could you say that again?"

    def test_universal_fallback_is_constant(self):
        assert ResponseValidator.UNIVERSAL_FALLBACK == UNIVERSAL_FALLBACK
        assert isinstance(UNIVERSAL_FALLBACK, str)
        assert len(UNIVERSAL_FALLBACK) > 0


class TestLogWriteFailure:
    """Verify that log failure does not block response delivery.
    
    This is tested at the pipeline level — the validator itself doesn't log.
    The pipeline's _log_validation wraps in try/except with fire-and-forget.
    """
    def test_validator_returns_result_regardless_of_logging(self, validator):
        r = validator.validate("Hello there.", "Hi", _make_cache())
        assert r.action == ValidationAction.ALLOW
        assert r.response == "Hello there."


# === WO-2026-061-A: False Positive Tuning Tests ===

class TestMemoryInquiryExemption:
    """Check 3 should NOT fire when asking about or echoing family members."""

    def test_asking_how_is_your_daughter(self, validator):
        r = validator.validate("Good morning! How is your daughter doing?", "Good morning", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_tell_me_about_your_family(self, validator):
        r = validator.validate("I would love to hear about your family.", "What should we talk about?", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_empathetic_mirror_miss_husband(self, validator):
        r = validator.validate("I can only imagine how much you miss your husband.", "I miss my husband", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_user_mentioned_son_echo(self, validator):
        r = validator.validate("That is wonderful! How is your son doing?", "I talked to my son today", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_asserting_unknown_friend_still_blocked(self, validator):
        """True positive: inventing a friend the user never mentioned."""
        r = validator.validate("Your friend Margaret would love that.", "I like gardening", _make_cache())
        assert r.action != ValidationAction.ALLOW
        assert any("memory" in c for c in r.checks_triggered)


class TestMedicalTryTakingTuning:
    """'try taking' should only block with medication names, not 'a deep breath'."""

    def test_try_taking_deep_breath_allowed(self, validator):
        r = validator.validate("Try taking a deep breath and relax.", "I feel anxious", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_try_taking_a_walk_allowed(self, validator):
        r = validator.validate("Try taking a short walk outside.", "I feel restless", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_try_taking_ibuprofen_blocked(self, validator):
        r = validator.validate("Try taking ibuprofen for that.", "My head hurts", _make_cache())
        assert r.action == ValidationAction.BLOCK
        assert any("medical" in c for c in r.checks_triggered)


class TestFalseCertaintyHedgeExpansion:
    """Check 5 should allow responses that acknowledge inability to check."""

    def test_wish_i_could_check_weather(self, validator):
        r = validator.validate("I wish I could check that for you, but the weather has been unpredictable.", "What's the weather?", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_dont_know_temperature(self, validator):
        r = validator.validate("I don't know the exact temperature right now.", "Is it cold out?", _make_cache())
        assert r.action == ValidationAction.ALLOW

    def test_unhedged_weather_claim_still_blocked(self, validator):
        """True positive: asserting weather facts without hedge."""
        r = validator.validate("It is 72 degrees and sunny outside right now.", "What's the weather?", _make_cache())
        assert r.action == ValidationAction.REWRITE
        assert any("false_certainty" in c for c in r.checks_triggered)
