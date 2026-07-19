"""
Phase 2, Feature 3 — the generator (online half of RAG, step 2).

Turns a question + retrieved chunks into a grounded, cited answer. The system prompt
does the safety-critical work: answer ONLY from the numbered context, cite as [n],
refuse when the context is insufficient, and never give personal medical advice.

Talks to any OpenAI-compatible endpoint (Ollama now, Groq/hosted later) — switching
provider is a config change (base_url + model), not a code change.

Run (quick test):  python -m rag.generator "what is the sinoatrial node"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from openai import OpenAI

from common import load_config
from rag.retriever import Hit, Retriever

# The exact string the model must emit when it cannot answer. Defined once here and
# interpolated into the prompt, because evaluation/run_generation_eval.py matches on it
# to measure refusal rate — two copies would silently drift apart.
REFUSAL_TEXT = "I don't have enough context to answer that."

SYSTEM_PROMPT = (
    "You are a medical-education tutor for students. Answer using ONLY the numbered "
    "context passages below. "
    f'If the context does not contain the answer, reply exactly: "{REFUSAL_TEXT}" '
    "Do not use outside knowledge or invent facts. "
    "This is educational content, not medical advice: do not diagnose, recommend "
    "treatment or drug doses, or interpret anyone's personal results.\n\n"
    # Small models drop citations entirely, or misread [n] as a section header and
    # dump passages verbatim underneath. Show the target shape; forbid both failures.
    "CITATIONS — write your own prose. EVERY sentence must end with at least one "
    "[n] naming the passages it draws on. The example below is only a formatting "
    "illustration; ignore its subject matter:\n"
    "  Compact bone forms the dense outer shell of the shaft [1]. Its osteons run "
    "parallel to the long axis, which is what resists bending loads [1][2].\n"
    "Put the [n] at the END of the sentence — do not write 'Figure 3 from [2]' or "
    "otherwise fold the number into the prose. Never begin a line with [n], never "
    "use [n] as a heading, and never quote a passage verbatim."
)


@dataclass
class Answer:
    text: str
    hits: list[Hit]  # the context the model was given, for display + citation lookup
    gated: bool = False  # True -> abstained on weak retrieval, no LLM call was made


class Generator:
    """Wraps an OpenAI-compatible chat model; produces grounded answers from context."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.llm = cfg["llm"]
        # Abstain when the best hit is weaker than this. 0.0 disables. See config comment
        # for the calibration; the gate lives here rather than in the retriever so hit@k
        # keeps measuring raw retrieval.
        self.min_score = cfg.get("retrieval", {}).get("min_score", 0.0)
        # Ollama ignores the key; a real value is only needed for hosted providers.
        api_key = os.environ.get(self.llm.get("api_key_env", ""), "") or "ollama"
        self.client = OpenAI(base_url=self.llm["base_url"], api_key=api_key)

    @staticmethod
    def _format_context(hits: list[Hit]) -> str:
        # Number passages [1..k] so the model's [n] citations map back to hits.
        return "\n\n".join(f"[{i}] ({h.source}) {h.title}\n{h.text}" for i, h in enumerate(hits, 1))

    def generate(self, query: str, hits: list[Hit]) -> Answer:
        # Hits arrive sorted by score, so hits[0] is the best match. Gating on it is both
        # cheaper and more reliable than asking the model to notice weak context: measured,
        # the model answers from memorised knowledge (Gleason, TNM, guidelines) regardless of
        # what the context holds. Skipping the call entirely is the point.
        if not hits or hits[0].score < self.min_score:
            return Answer(text=REFUSAL_TEXT, hits=hits, gated=True)

        user = f"Context:\n{self._format_context(hits)}\n\nQuestion: {query}"
        resp = self.client.chat.completions.create(
            model=self.llm["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=self.llm.get("temperature", 0.1),
            max_tokens=self.llm.get("max_tokens", 700),
        )
        return Answer(text=resp.choices[0].message.content.strip(), hits=hits)


def main() -> int:
    query = " ".join(sys.argv[1:]) or "what is the sinoatrial node"
    cfg = load_config()
    with Retriever(cfg) as r:
        hits = r.retrieve(query, k=cfg["llm"].get("top_k", 5))
    answer = Generator(cfg).generate(query, hits)
    print(f"Q: {query}\n")
    print(answer.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
