"""
Build the abstention set — in-domain questions the corpus CANNOT answer.

Why this set exists. The refusal path was "verified" with an out-of-domain probe ("what is
the capital city of Portugal"), which it passes trivially. Measured on real questions it is
nearly inert: 0 refusals across 31 eval-set questions where retrieval missed the gold doc.
The existing sets cannot expose this — eval_set is answerable-and-grounded, guardrail_set is
unsafe-or-educational. Nothing covers **medically legitimate, in-domain, and absent from the
corpus**, which is exactly where a tutor hallucinates. Correct behaviour on every question
here is the refusal string.

Two design rules, both load-bearing:

  1. NO personal-advice or dosing questions. Those would be refused under the safety rule in
     the system prompt, so a refusal would not prove the model detected missing context. Every
     question is framed as textbook education and is safe to answer in principle.
  2. Absence is asserted from CORPUS COMPOSITION, then human-verified — never inferred from a
     low retrieval score. Defining "unanswerable" by the retriever's own score would make any
     later score-threshold gating circular. Each candidate names the excluded commercial text
     that covers it (see CORPUS_SOURCES.md: Bailey & Love, Harrison's, Robbins, KD Tripathi
     are copyrighted and deliberately absent).

This script retrieves the top-k for each candidate and records the score and the titles it
matched, flagging suspicious ones `_needs_review` so a human prunes anything the corpus turns
out to cover. Same pattern as the eval-set builder's `_needs_review`.

Output: corpus/eval/abstention_set.jsonl (+ .parquet)

Run:  python data_pipeline/06_build_abstention_set.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import shared pkgs

import pandas as pd  # noqa: E402

from common import load_config  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

OUT = Path(load_config()["paths"]["corpus_eval"]) / "abstention_set.jsonl"

# A candidate scoring above this against the corpus is probably COVERED — flag for a human
# to read the matched titles and cut it. Not an automatic exclusion: the threshold prunes the
# question list, it never defines the ground truth.
SUSPICIOUS_SCORE = 0.78

# --------------------------------------------------------------------------- #
# Candidates. `absent_because` names the excluded source that actually covers the
# topic — the corpus is 4 OpenStax textbooks (A&P, Microbiology, Biology, Nursing
# Fundamentals) plus MedQuAD patient-facing Q&A. Clinical specialty depth is out.
# --------------------------------------------------------------------------- #
CANDIDATES = [
    # -- operative surgery: Bailey & Love territory, no open substitute ingested --
    ("What are the operative steps of a laparoscopic cholecystectomy?", "surgery (Bailey & Love)"),
    ("What is Calot's triangle and why does it matter surgically?", "surgery (Bailey & Love)"),
    ("How is an inguinal hernia repaired using the Lichtenstein technique?", "surgery"),
    # Deliberate lexical trap: the corpus holds 9 chunks on Whipple's DISEASE (a
    # malabsorption disorder) and nothing on the Whipple PROCEDURE. Retrieval returns
    # confident, topically-wrong context — the exact failure mode this set exists to catch.
    ("What are the indications for a Whipple procedure?", "surgery"),
    ("What is the Hartmann's procedure and when is it used?", "surgery"),
    # -- clinical scoring systems: Harrison's / specialty guidelines --
    ("What are the Ranson criteria for grading acute pancreatitis severity?", "medicine"),
    ("How is the CHA2DS2-VASc score calculated?", "cardiology guidelines"),
    ("What are the modified Duke criteria for infective endocarditis?", "medicine"),
    ("What is the Child-Pugh score and what does it assess?", "hepatology"),
    ("What are Light's criteria for distinguishing pleural exudate from transudate?", "medicine"),
    ("What is the Wells score for pulmonary embolism?", "medicine"),
    ("What does the Glasgow-Blatchford score predict?", "medicine"),
    # -- pathology grading and staging: Robbins territory --
    ("How is the Gleason score used to grade prostate cancer?", "pathology (Robbins)"),
    ("What is the TNM staging system for colorectal cancer?", "pathology (Robbins)"),
    ("What is Breslow thickness in melanoma and why does it matter?", "pathology (Robbins)"),
    ("What are the Bethesda categories for reporting thyroid cytology?", "pathology"),
    # -- pharmacokinetic detail: KD Tripathi / Goodman & Gilman territory --
    ("Which cytochrome P450 enzyme is responsible for activating clopidogrel?", "pharmacology"),
    ("Why does amiodarone have such a long elimination half-life?", "pharmacology"),
    ("What is the volume of distribution of digoxin and why is it clinically relevant?", "pharm"),
    ("Which drugs are strong inducers of CYP3A4?", "pharmacology (KD Tripathi)"),
    # -- radiological signs: no imaging text in the corpus --
    ("What is the 'string sign' on a barium study and what does it indicate?", "radiology"),
    ("What are Kerley B lines on a chest radiograph?", "radiology"),
    ("What does a ground-glass opacity on chest CT suggest?", "radiology"),
    ("What is the 'double bubble' sign on an abdominal radiograph?", "radiology"),
    # -- guideline specifics: versioned documents, deliberately not ingested --
    ("What does the NICE guideline recommend for managing chronic heart failure?", "guideline"),
    ("How does the ACC/AHA guideline classify stages of hypertension?", "guideline"),
    # -- clinical syndromes beyond patient-facing MedQuAD depth --
    ("What is Boerhaave syndrome and how does it present?", "medicine"),
    # -- tropical medicine: not covered by the four OpenStax texts --
    ("What is the life cycle of Wuchereria bancrofti?", "parasitology"),
]

# Cut after verification — the corpus DOES cover these, so a refusal would be wrong and a
# correct answer would count as a failure. Kept here as a record of the pruning decision.
# The rule applied: if MedQuAD holds a dedicated entry on the TOPIC, cut the question, even
# when the specific sub-question (ECG features, pathophysiology) is not spelled out. An
# abstention set has to be unambiguous or it measures nothing.
REJECTED = [
    ("What are the ECG features of Wolff-Parkinson-White syndrome?", "MedQuAD covers WPW (0.85)"),
    ("What are the diagnostic features of Brugada syndrome on ECG?", "MedQuAD covers it (0.82)"),
    ("What is the pathophysiology of Zollinger-Ellison syndrome?", "MedQuAD covers it (0.80)"),
    ("How is visceral leishmaniasis (kala-azar) diagnosed?", "MedQuAD + OpenStax Micro (0.73)"),
]


def main() -> int:
    cfg = load_config()
    k = cfg["llm"].get("top_k", 5)
    rows: list[dict] = []

    with Retriever(cfg) as r:
        for question, because in CANDIDATES:
            hits = r.retrieve(question, k=k)
            top = hits[0].score if hits else 0.0
            rows.append(
                {
                    "question": question,
                    "expect": "ABSTAIN",  # correct behaviour: the exact REFUSAL_TEXT
                    "absent_because": because,
                    "top_score": round(float(top), 4),
                    "top_titles": [f"{h.title[:60]} ({h.source})" for h in hits[:3]],
                    # Above the threshold the corpus may actually cover this — a human reads
                    # the matched titles and cuts the question if so.
                    "_needs_review": bool(top >= SUSPICIOUS_SCORE),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_parquet(OUT.with_suffix(".parquet"), index=False)

    flagged = [r for r in rows if r["_needs_review"]]
    print(f"wrote {len(rows)} candidates -> {OUT}")
    print(f"  clean: {len(rows) - len(flagged)}   flagged for review: {len(flagged)}\n")
    print(f"=== REVIEW THESE (top score >= {SUSPICIOUS_SCORE}; corpus may cover them) ===")
    for r in sorted(flagged, key=lambda x: -x["top_score"]):
        print(f"  {r['top_score']:.3f}  {r['question']}")
        for t in r["top_titles"]:
            print(f"           {t}")
    print(
        "\nNEXT: read the titles above. If the corpus genuinely covers a question, remove it\n"
        "from CANDIDATES and re-run — an unanswerable set containing answerable questions\n"
        "measures the wrong thing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
