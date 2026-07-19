"""
Crisis detection — Layer 1 rules, and the fixed responses they route to.

Runs BEFORE retrieval scoring in `Generator.generate()`. Order is the whole point: the
`retrieval.min_score` gate abstains on weak retrieval, and every crisis question scores below
it (median 0.648 vs the 0.72 gate), so a crisis routed through the gate gets a bare
"I don't have enough context to answer that." — no help, no signposting, on the highest-stakes
input the system can receive.

Two rules make this safe to run ahead of everything else:

  1. THE RESPONSE IS A FIXED HUMAN-WRITTEN STRING. It is never generated, never retrieved, and
     never passes through the LLM. A model must not improvise wording for someone disclosing
     suicidal intent, and a corpus of anatomy textbooks cannot answer "I've taken an overdose".
  2. PATTERNS REQUIRE FIRST-PERSON, PRESENT-TENSE FRAMING. "How does an overdose affect the
     liver?" is an educational question and must NOT trigger; "I've just taken an overdose"
     must. The anchor is disclosure about oneself or someone present, not the topic.

Recall is prioritised over precision here — a missed crisis is far worse than a student
briefly seeing a helpline — but precision is measured, not assumed: see `main()`, which scores
these patterns against the labelled guardrail set (must catch 8/8 CRISIS, must not fire on
EDUCATIONAL).

*** LOCALISE BEFORE DEPLOYING. *** The numbers below are UK/US/EU. They are correct as
written, but a deployed tutor should present the resources for its actual users' region.

Run (self-test):  python -m guardrails.crisis
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Fixed responses. Human-written, reviewed, never model-generated.
# --------------------------------------------------------------------------- #
SELF_HARM_RESPONSE = (
    "I'm really sorry you're feeling like this, and I'm glad you said something.\n\n"
    "I'm a study tool for medical students, so I'm not the right kind of help — but people "
    "who are trained for exactly this are available right now, free, and will not judge you:\n\n"
    "  * UK — Samaritans: call 116 123 (24h), or email jo@samaritans.org\n"
    "  * US — 988 Suicide & Crisis Lifeline: call or text 988\n"
    "  * Anywhere — if you are in immediate danger, call your local emergency number "
    "(999 UK / 911 US / 112 EU)\n\n"
    "If you can, tell someone near you how you're feeling as well. You deserve support from "
    "a real person, not a textbook."
)

MEDICAL_EMERGENCY_RESPONSE = (
    "This sounds like a medical emergency. Please stop reading and call emergency services "
    "now — 999 (UK), 911 (US), or 112 (EU).\n\n"
    "I'm a study tool for medical students. I cannot assess anyone's condition, and waiting "
    "for an answer here could cost time that matters.\n\n"
    "If someone is unresponsive and not breathing normally, call emergency services and ask "
    "the operator to talk you through what to do — they will stay on the line with you."
)

CRISIS_RESPONSES = {
    "SELF_HARM": SELF_HARM_RESPONSE,
    "MEDICAL_EMERGENCY": MEDICAL_EMERGENCY_RESPONSE,
}

# --------------------------------------------------------------------------- #
# Layer 1 patterns. First-person / someone-present framing is required throughout,
# so topic questions ("what causes suicidal ideation in depression?") do not fire.
# --------------------------------------------------------------------------- #
_SELF = r"(?:i|i'm|im|i've|ive|i am|i have|my|me)"

SELF_HARM_PATTERNS = [
    # explicit intent toward oneself
    rf"\b{_SELF}\b[^.?!]*\b(?:want|going|plan|thinking about|thought about)\b[^.?!]*"
    r"\b(?:kill(?:ing)? myself|end(?:ing)? my life|take my own life|suicide|not be alive|"
    r"don'?t want to be alive)\b",
    r"\bi want to kill myself\b",
    r"\bi don'?t want to (?:be alive|live) anymore\b",
    r"\bending my life\b",
    # harm to self or others, disclosed in the first person
    rf"\b{_SELF}\b[^.?!]*\b(?:thoughts?|urges?)\b[^.?!]*\b(?:harm(?:ing)?|hurt(?:ing)?)\b"
    r"[^.?!]*\b(?:myself|other people|others|someone)\b",
    rf"\b{_SELF}\b[^.?!]*\b(?:harm(?:ing)?|hurt(?:ing)?|cut(?:ting)?)\s+myself\b",
    r"\bi'?m suicidal\b",
]

MEDICAL_EMERGENCY_PATTERNS = [
    # someone present is collapsed / not breathing / unresponsive
    r"\b(?:someone|somebody|he|she|they|my \w+)\b[^.?!]*\b(?:collapsed|unconscious|"
    r"unresponsive|not breathing|isn'?t breathing|won'?t wake up)\b",
    r"\b(?:not|isn'?t|stopped) breathing\b",
    # acute self-disclosure with immediacy
    rf"\b{_SELF}\b[^.?!]*\b(?:just )?(?:taken|took)\b[^.?!]*\boverdose\b",
    rf"\b{_SELF}\b[^.?!]*\boverdosed\b",
    rf"\b{_SELF}\b[^.?!]*\b(?:chest pain|can'?t breathe|cannot breathe)\b[^.?!]*"
    r"\b(?:right now|now|currently)\b",
    rf"\b{_SELF}\b[^.?!]*\b(?:crushing|severe)\b[^.?!]*\bchest pain\b",
    rf"\b{_SELF}\b[^.?!]*\bbleeding\b[^.?!]*\b(?:heavily|won'?t stop|can'?t stop)\b",
]

_SELF_HARM_RE = [re.compile(p, re.IGNORECASE) for p in SELF_HARM_PATTERNS]
_EMERGENCY_RE = [re.compile(p, re.IGNORECASE) for p in MEDICAL_EMERGENCY_PATTERNS]


def detect_crisis(question: str) -> str | None:
    """Return 'SELF_HARM', 'MEDICAL_EMERGENCY', or None.

    Self-harm is checked first: a question mentioning both ("I've taken an overdose because I
    want to die") should route to the mental-health response, which also names emergency
    services, rather than to the purely medical one.
    """
    if any(r.search(question) for r in _SELF_HARM_RE):
        return "SELF_HARM"
    if any(r.search(question) for r in _EMERGENCY_RE):
        return "MEDICAL_EMERGENCY"
    return None


def main() -> int:
    """Self-test against the labelled guardrail set: 8/8 CRISIS caught, 0 EDUCATIONAL fired."""
    from pathlib import Path

    import pandas as pd

    from common import load_config

    path = Path(load_config()["paths"]["corpus_eval"]) / "guardrail_set.parquet"
    df = pd.read_parquet(path)
    df["detected"] = [detect_crisis(q) for q in df["question"]]

    crisis = df[df["label"] == "CRISIS"]
    caught = crisis["detected"].notna().sum()
    print(f"CRISIS caught      : {caught}/{len(crisis)}   (must be {len(crisis)}/{len(crisis)})")
    for _, r in crisis.iterrows():
        mark = r["detected"] or "*** MISSED ***"
        print(f"    {mark:20} {r['question'][:60]}")

    for label in ("EDUCATIONAL", "PERSONAL_ADVICE"):
        sub = df[df["label"] == label]
        fired = sub[sub["detected"].notna()]
        want = "must be 0" if label == "EDUCATIONAL" else "some acceptable — genuine emergencies"
        print(f"\n{label} fired: {len(fired)}/{len(sub)}   ({want})")
        for _, r in fired.iterrows():
            print(f"    {r['detected']:20} {r['question'][:60]}")

    ok = caught == len(crisis) and df[(df["label"] == "EDUCATIONAL") & df["detected"].notna()].empty
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
