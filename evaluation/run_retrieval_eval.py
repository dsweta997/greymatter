"""
Phase 3, step 1 — baseline retrieval eval (judge-free, deterministic).

Runs the dense retriever over the grounded eval set and reports hit-rate@k and MRR.
This is the "before" number the Phase 3 hybrid + reranker work is measured against.

This script MEASURES (does not generate data), so it lives in evaluation/, per
markdown/CODING_STANDARDS.md.

Run:  python evaluation/run_retrieval_eval.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import shared pkgs

import pandas as pd  # noqa: E402

from common import load_config  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

KS = (1, 3, 5, 10)


def main() -> int:
    cfg = load_config()
    eval_path = Path(cfg["paths"]["corpus_eval"]) / "eval_set.parquet"
    df = pd.read_parquet(eval_path)

    # Score only questions with TRUSTWORTHY ground truth. The 10 cross-source questions
    # are _needs_review — keyword grounding can't reliably pin their multi-doc answers, so
    # their doc-ids are provisional. Skip them; don't let unreliable labels poison the number.
    graded = df[~df["_needs_review"].astype(bool)].reset_index(drop=True)
    questions = graded["question"].tolist()
    print(
        f"eval set: {len(df)} questions | {len(graded)} grounded (scored) | "
        f"{len(df) - len(graded)} skipped (_needs_review)"
    )

    # Retrieve top-K once for every question (batch-encoded).
    with Retriever() as r:
        results = r.retrieve_many(questions, k=max(KS))

    # rank of the first relevant doc in each question's results (1-based), or None.
    ranks: list[int | None] = []
    for hits, gold in zip(results, graded["relevant_doc_ids"]):
        gold = set(gold)
        ranks.append(next((j for j, h in enumerate(hits, 1) if h.doc_id in gold), None))

    n = len(ranks)
    print("\n=== overall ===")
    for k in KS:
        hit = sum(1 for rank in ranks if rank is not None and rank <= k) / n
        print(f"  hit@{k:<2} = {hit:.3f}")
    mrr = sum(1 / rank for rank in ranks if rank is not None) / n
    print(f"  MRR@{max(KS)} = {mrr:.3f}")

    # hit@5 broken down by subject and by question type (diagnoses where retrieval is weak).
    for field in ("subject", "type"):
        print(f"\n=== hit@5 by {field} ===")
        buckets: dict[str, list[bool]] = defaultdict(list)
        for rank, key in zip(ranks, graded[field]):
            buckets[key].append(rank is not None and rank <= 5)
        for key, hits in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            print(f"  {key:20} {sum(hits) / len(hits):.3f}  (n={len(hits)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
