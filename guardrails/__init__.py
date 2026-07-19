"""
Safety guardrails (proposal Phase 3.5), built cheapest-layer-first.

Present: `crisis` — Layer 1 rules for crisis detection, pulled forward ahead of the rest of
Phase 3.5 because the answer-time score gate (`retrieval.min_score`) otherwise routes every
crisis question to a bare "I don't have enough context" with no signposting. All 8 crisis
questions in the guardrail set score below the gate, so without this the system produces its
worst possible response to its highest-stakes input.

To come (Phase 3.5 proper): the EDUCATIONAL / PERSONAL_ADVICE classifier, rules then LLM
few-shot, measured by confusion matrix over the full guardrail set.
"""

from guardrails.crisis import CRISIS_RESPONSES, detect_crisis

__all__ = ["CRISIS_RESPONSES", "detect_crisis"]
