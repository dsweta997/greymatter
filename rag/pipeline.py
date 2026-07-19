"""
Phase 2, Feature 4 — the RAG pipeline CLI.

Wires retriever + generator end-to-end and prints BOTH the retrieved context and the
answer. Always showing what the model saw is what makes the failure catalogue possible:
if the right chunk isn't in the context it's a RETRIEVAL bug; if it is there but the
answer is wrong it's a GENERATION bug.

Run:  python -m rag.pipeline "why do we have two circulatory loops"
"""

from __future__ import annotations

import sys

from common import load_config
from rag.generator import Generator
from rag.retriever import Retriever


def main() -> int:
    query = " ".join(sys.argv[1:])
    if not query:
        print('usage: python -m rag.pipeline "your question"')
        return 1

    cfg = load_config()
    k = cfg["llm"].get("top_k", 5)
    with Retriever(cfg) as retriever:
        hits = retriever.retrieve(query, k=k)
        answer = Generator(cfg).generate(query, hits)

    print(f"\nQUESTION: {query}\n")
    print("RETRIEVED CONTEXT (what the model saw):")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.score:.3f} ({h.source}) {h.title[:55]}")
        print(f"      {h.text[:150].strip()}...")

    print("\nANSWER:")
    if answer.gated:
        # Distinguish "the system declined to try" from "the model read the context and
        # declined" — they have different causes and different fixes.
        best = hits[0].score if hits else 0.0
        print(f"  [gated: best hit {best:.3f} < min_score; no LLM call made]")
    print(answer.text)

    print("\nSOURCES:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h.title} — {h.source} ({h.license})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
