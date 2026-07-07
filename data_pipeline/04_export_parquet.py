"""
Export the working datasets to Parquet — the committed / distributable form.

- corpus/raw/*.json  -> corpus/corpus.parquet         (24,922 docs in one file)
- corpus/eval/eval_set.jsonl      -> corpus/eval/eval_set.parquet
- corpus/eval/guardrail_set.jsonl -> corpus/eval/guardrail_set.parquet

The per-file JSON / JSONL stay as the human-editable, resumable working form; Parquet is
the derived artifact (smaller, faster, columnar, matches how MedQuAD ships). The `metadata`
bag is stored as a JSON string column so the schema stays stable across sources.

Run:  python data_pipeline/04_export_parquet.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import shared pkgs

from datasets import Dataset  # noqa: E402

from common import load_config  # noqa: E402

_CFG = load_config()
RAW = Path(_CFG["paths"]["corpus_raw"])
EVAL = Path(_CFG["paths"]["corpus_eval"])


def export_corpus():
    rows = []
    for f in glob.glob(str(RAW / "*.json")):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "id": d["id"],
                "title": d.get("title", ""),
                "text": d["text"],
                "source": d["source"],
                "license": d["license"],
                "attribution": d["attribution"],
                # stringify the variable metadata bag -> stable columnar schema
                "metadata": json.dumps(d.get("metadata", {}), ensure_ascii=False),
            }
        )
    out = Path("corpus/corpus.parquet")
    Dataset.from_list(rows).to_parquet(str(out))
    print(f"corpus:    {len(rows):6} docs -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def export_jsonl(name, keys_defaults):
    src = EVAL / f"{name}.jsonl"
    if not src.exists():
        print(f"{name}: {src} missing, skipping")
        return
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # normalize optional keys so every row shares one schema
        for k, default in keys_defaults.items():
            r.setdefault(k, default)
        rows.append(r)
    out = EVAL / f"{name}.parquet"
    Dataset.from_list(rows).to_parquet(str(out))
    print(f"{name}: {len(rows):6} rows -> {out}  ({out.stat().st_size / 1e3:.0f} KB)")


def main():
    export_corpus()
    export_jsonl(
        "eval_set",
        {
            "relevant_doc_ids": [],
            "must_contain": [],
            "type": "",
            "subject": "",
            "_needs_review": False,
        },
    )
    export_jsonl(
        "guardrail_set",
        {
            "label": "",
            "boundary": False,
            "rationale": "",
        },
    )


if __name__ == "__main__":
    main()
