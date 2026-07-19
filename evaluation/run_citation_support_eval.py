"""
Phase 3, step 3 — do citations carry information? (judge-free, deterministic)

`run_generation_eval.py` measures citation COVERAGE: does a sentence end in [n]. That
turned out to be a formatting metric — coverage was identical whether retrieval found the
gold document (0.75) or missed it (0.77), so the model cites just as diligently from the
wrong context. Coverage cannot tell us whether [n] was EARNED.

This script tests that directly, by comparing each cited sentence against three things:

  1. CITED     cos(sentence, the chunk it cited)
  2. OTHER     cos(sentence, the retrieved chunks it did NOT cite in that sentence)
  3. RANDOM    cos(sentence, a chunk drawn at random from the corpus)

Read the result like this:
  * CITED >> RANDOM but CITED ~= OTHER  -> citations are topically right but do not
    discriminate WITHIN the context: [2] and [4] were interchangeable. Decorative.
  * CITED ~= RANDOM                      -> citations are noise. The metric is dead.
  * CITED >  OTHER >  RANDOM             -> citations point at the specific supporting
    passage. Coverage is then worth reporting alongside this support score.

Also reports CITED RANK: where the cited chunk sits when the k retrieved chunks are ranked
by similarity to the sentence. Rank ~1 means the model cited the best-supporting passage;
rank ~k/2 means it cited at chance.

Uses the retriever's own bge-small model for scoring — no LLM judge, no new dependency, so
this stays judge-free like the other evals. It is an approximation: paraphrase lowers cosine
even when the passage genuinely supports the sentence, so treat CITED-vs-OTHER-vs-RANDOM as
a relative comparison, never as an absolute faithfulness score.

Answers are read from an existing --transcript rather than regenerated; retrieval is redone
(deterministic: same query, same index, same top-k) and cross-checked against the stored
retrieval_hit to prove the context was reconstructed faithfully.

This script MEASURES (does not generate data), so it lives in evaluation/, per
markdown/CODING_STANDARDS.md.

Run:  python evaluation/run_citation_support_eval.py --transcript evalset_7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import shared pkgs

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import load_config  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

MIN_SENTENCE_CHARS = 15
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CITATION = re.compile(r"\[(\d+)\]")
RANDOM_SEED = 0  # fixed so the random baseline is reproducible run to run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="jsonl from run_generation_eval.py --dataset eval",
    )
    return p.parse_args()


def sentences_with_citations(answer: str) -> list[tuple[str, list[int]]]:
    """(sentence, [cited indices]) for each scoreable sentence that cites at least one."""
    out = []
    for s in SENTENCE_SPLIT.split(answer):
        s = s.strip()
        if len(s) < MIN_SENTENCE_CHARS:
            continue
        idx = [int(n) for n in CITATION.findall(s)]
        if idx:
            out.append((s, sorted(set(idx))))
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config()
    k = cfg["llm"].get("top_k", 5)
    lines = args.transcript.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    print(f"transcript: {args.transcript.name} | records: {len(records)} | top_k: {k}\n")

    rng = np.random.default_rng(RANDOM_SEED)
    corpus = pd.read_parquet("corpus/corpus.parquet")["text"].tolist()

    cited_s: list[float] = []
    other_s: list[float] = []
    random_s: list[float] = []
    ranks: list[int] = []
    out_of_range = 0

    with Retriever(cfg) as r:
        model = r.model  # same bge-small the index was built with
        for rec in records:
            # Retrieval is deterministic (same query, same frozen index, same top-k), so this
            # reconstructs exactly the context the generation run passed to the model. If the
            # index is ever rebuilt, transcripts predating the rebuild become unscoreable here.
            hits = r.retrieve(rec["question"], k=k)
            pairs = sentences_with_citations(rec["answer"])
            if not pairs:
                continue

            sent_vecs = model.encode([s for s, _ in pairs], normalize_embeddings=True)
            chunk_vecs = model.encode([h.text for h in hits], normalize_embeddings=True)
            rand_vecs = model.encode(
                [corpus[i] for i in rng.integers(0, len(corpus), len(pairs))],
                normalize_embeddings=True,
            )

            for i, (_, idx) in enumerate(pairs):
                sims = chunk_vecs @ sent_vecs[i]  # cosine: vectors are normalized
                valid = [n for n in idx if 1 <= n <= len(hits)]
                out_of_range += len(idx) - len(valid)
                if not valid:
                    continue
                cited_pos = [n - 1 for n in valid]
                others = [j for j in range(len(hits)) if j not in cited_pos]

                cited_s.append(float(np.mean([sims[j] for j in cited_pos])))
                if others:
                    other_s.append(float(np.mean([sims[j] for j in others])))
                random_s.append(float(rand_vecs[i] @ sent_vecs[i]))
                # Rank of the best cited chunk among all k, 1 = best-supporting.
                order = list(np.argsort(-sims))
                ranks.append(min(order.index(j) + 1 for j in cited_pos))

    n = len(cited_s)
    if not n:
        print("no cited sentences found — nothing to score")
        return 1

    print(f"=== citation support (n={n} cited sentences) ===")
    print(f"  CITED  chunk similarity : {np.mean(cited_s):.3f}")
    print(f"  OTHER  retrieved chunks : {np.mean(other_s):.3f}")
    print(f"  RANDOM corpus chunk     : {np.mean(random_s):.3f}   <- floor")
    print(f"\n  lift over random  : {np.mean(cited_s) - np.mean(random_s):+.3f}")
    print(f"  lift over other   : {np.mean(cited_s) - np.mean(other_s):+.3f}   <- the real test")

    chance = (k + 1) / 2
    print(f"\n=== cited rank among {k} retrieved (1 = best-supporting) ===")
    print(f"  mean rank : {np.mean(ranks):.2f}   (chance = {chance:.2f})")
    for rank in range(1, k + 1):
        share = sum(1 for x in ranks if x == rank) / len(ranks)
        print(f"    rank {rank}: {share:5.1%}")

    if out_of_range:
        print(f"\n  WARNING: {out_of_range} citations pointed outside [1..{k}] (fabricated index)")

    print(
        "\nINTERPRETATION\n"
        "  lift over other ~ 0  -> citations are topically plausible but interchangeable:\n"
        "                          [2] and [4] would have scored the same. Decorative.\n"
        "  lift over random ~ 0 -> citations carry no information at all.\n"
        "  both clearly > 0     -> citations point at the specific supporting passage.\n"
        "  Cosine under-rewards paraphrase, so read these as a RELATIVE comparison only —\n"
        "  this is not a faithfulness score."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
