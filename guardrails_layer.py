"""
Phase 4 — Guardrails Layer
Input and output validation for the Travel Concierge agent.

Architecture:
  User input → [INPUT GUARDS] → LLM + Tools → [OUTPUT GUARDS] → User
"""

import re
from dataclasses import dataclass
from typing import Optional


# ── Result type ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed: bool
    value: str                        # original or fixed text
    error_message: Optional[str] = None
    guard_name: Optional[str]   = None


# ── Input Guard 1: Toxic language + prompt injection ─────────────────────

TOXIC_PATTERNS = [
    r"\b(fuck|shit|bitch|asshole|bastard|cunt)\b",
    r"ignore (your|all|previous) (instructions?|prompts?|rules?)",
    r"(you are now|act as|pretend (you are|to be)|roleplay as)",
    r"(jailbreak|bypass|override|system prompt)",
    r"(drop table|union select|exec\(|<script)",
]

def guard_toxic_input(text: str) -> ValidationResult:
    """Block abusive language and prompt injection attempts."""
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ValidationResult(
                passed=False,
                value=text,
                error_message="I can't process that message. Please keep it respectful and travel-related.",
                guard_name="ToxicInputGuard",
            )
    return ValidationResult(passed=True, value=text)


# ── Input Guard 2: On-topic check ────────────────────────────────────────

TRAVEL_KEYWORDS = [
    "stay", "hotel", "airbnb", "room", "house", "apartment", "villa",
    "cabin", "listing", "book", "night", "pool", "beach", "downtown",
    "accommodate", "guest", "property", "bedroom", "bathroom", "amenities",
    "review", "rating", "rent", "vacation", "trip", "travel", "holiday",
    "resort", "place", "find", "looking", "need", "want", "search",
    "cheap", "budget", "luxury", "ocean", "mountain", "city", "quiet",
    "cozy", "spacious", "pet", "parking", "wifi", "kitchen",
]

def guard_on_topic(text: str) -> ValidationResult:
    """Redirect queries unrelated to travel and accommodation."""
    text_lower = text.lower()

    # Short greetings pass through
    if len(text.split()) <= 5 and any(g in text_lower for g in ["hi", "hello", "hey", "help", "thanks"]):
        return ValidationResult(passed=True, value=text)

    if any(kw in text_lower for kw in TRAVEL_KEYWORDS):
        return ValidationResult(passed=True, value=text)

    return ValidationResult(
        passed=False,
        value=text,
        error_message=(
            "I'm a travel concierge — I can only help with accommodation searches. "
            "Try: 'Find me a beachfront villa for 4 people under $300/night'."
        ),
        guard_name="OnTopicGuard",
    )


# ── Output Guard 1: Competitor mention filter ─────────────────────────────

COMPETITORS = [
    "booking.com", "hotels.com", "expedia", "vrbo", "tripadvisor",
    "trivago", "kayak", "agoda", "hostelworld", "homeaway",
]

def guard_competitor_mentions(text: str) -> ValidationResult:
    """Remove sentences that mention competitor booking platforms."""
    found = [c for c in COMPETITORS if c in text.lower()]

    if not found:
        return ValidationResult(passed=True, value=text)

    # Remove any sentence containing a competitor name
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned   = [s for s in sentences if not any(c in s.lower() for c in found)]
    fixed     = " ".join(cleaned).strip()

    return ValidationResult(
        passed=True,          # pass but return fixed version
        value=fixed,
        error_message=f"Removed competitor mention(s): {found}",
        guard_name="CompetitorGuard",
    )


# ── Guard orchestrator ────────────────────────────────────────────────────

class TravelConciergeGuard:
    """
    Wraps the agent call with input and output validation.

    Usage:
        guard = TravelConciergeGuard()

        passed, value = guard.validate_input(user_message)
        if not passed:
            print(value)   # show error to user, skip LLM
            return

        llm_response = agent.invoke(...)
        clean_response = guard.validate_output(llm_response)
        print(clean_response)
    """

    def validate_input(self, text: str) -> tuple[bool, str]:
        """
        Run all input guards in order.
        Returns (True, cleaned_text) or (False, error_message_for_user).
        """
        for validator in [guard_toxic_input, guard_on_topic]:
            result = validator(text)
            if not result.passed:
                print(f"  ⛔ [GUARD BLOCKED] {result.guard_name}")
                return False, result.error_message
        return True, text

    def validate_output(self, text: str) -> str:
        """
        Run all output guards.
        Returns cleaned response text.
        """
        result = guard_competitor_mentions(text)
        if result.error_message:
            print(f"  ✂️  [GUARD FIXED]   {result.guard_name}: {result.error_message}")
        return result.value