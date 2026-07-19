# Gray Matter

A medical-education RAG tutor: grounded, cited answers built only from openly-licensed
medical textbooks. Runs entirely on a local GPU — no API keys, no cloud, no per-query cost.

> **Education, not advice.** A study tool for anatomy/physiology *concepts* — never for
> diagnosis, dosing, or interpreting personal results.
>
> **Not yet safe to hand to a student.** Personal-health questions are refused only 41% of the
> time (target ≥95%); the classifier that fixes this is Phase 3.5 and isn't built. Crisis
> questions *are* handled (fixed human-written signposting, 8/8, never model-generated).

## Quick start

Needs Python 3.11 and [Ollama](https://ollama.com). A CUDA GPU is optional but turns a
30-minute index build into ~90 seconds (built on an 8 GB RTX 4060). No `.env` required —
Ollama and embedded Qdrant don't authenticate.

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
ollama pull qwen2.5:7b

# Corpus + index are gitignored. Rebuild once (see note below for the PDFs):
python data_pipeline/01_ingest_corpus.py
python data_pipeline/04_export_parquet.py
python data_pipeline/05_embed_and_index.py

python -m rag.pipeline "why do we have two circulatory loops"
```

The four OpenStax PDFs aren't in the repo. Download them from
`openstax.org/details/books/<slug>` (slugs in `sources.openstax_books` in
[configs/config.yaml](configs/config.yaml)) into `corpus/source_files/`. MedQuAD pulls
automatically at a pinned revision.

## How a question flows

```
question → crisis check → retrieve top-5 → min_score gate → generate cited answer
             │                                    │
             └─ fixed response, no LLM            └─ abstain, no LLM
```

Three decisions worth knowing: crisis is checked **first** (those questions score low, so any
retrieval gate would swallow them); the score gate lives in the **generator, not the
retriever** (gating in `retrieve()` would silently change hit@k); and the LLM provider is a
**config value** (`llm.base_url` + `llm.model`), never a code change.

## Layout

| path | holds |
|---|---|
| `data_pipeline/` | numbered scripts that **generate data**, run in order |
| `rag/` | query-time runtime: retriever, generator, pipeline CLI |
| `guardrails/` | safety layer — crisis rules now, classifier to come |
| `evaluation/` | scripts that **measure** and never write data |
| `common/` | config loading, device selection, Qdrant client |
| `configs/config.yaml` | every setting; nothing hardcoded elsewhere |

## Evaluations

```bash
python evaluation/run_retrieval_eval.py                                   # hit@k + MRR (no LLM)
python evaluation/run_generation_eval.py --dataset guardrail --all        # does it refuse?
python evaluation/run_generation_eval.py --dataset eval --all             # does it cite?
python evaluation/run_generation_eval.py --dataset abstention --all       # does it admit ignorance?
python evaluation/run_citation_support_eval.py --transcript out.jsonl     # are the [n] earned?
```

Add `--model qwen2.5:3b` to compare models; omit `--all` for a fast sample. **Read a
`--transcript` before trusting any number** — two metrics here looked healthy while measuring
the wrong thing (`markdown/LEARNINGS.md`).

## Status

Phase 1 (corpus + eval sets) and Phase 2 (baseline RAG) complete. Phase 3 next: hybrid
retrieval + reranking, then the Phase 3.5 classifier.

| measurement | value | target |
|---|---|---|
| retrieval hit@5 | 0.656 | ≥0.80 |
| crisis redirect | 8/8 | 8/8 ✅ |
| educational wrongly refused | 3% | ≤5% ✅ |
| personal advice refused | 41% | ≥95% ❌ |
| abstention on unanswerable | 0.93 | 1.00 |
| citation coverage | 0.76 | *formatting metric — see below* |
| faithfulness | not measured | ≥0.80 |

Citation coverage scores the same whether or not retrieval found the right document, so it
measures whether `[n]` appears, not whether the claim is supported. The citations themselves
are accurate (cited chunk is the best match 74% of the time vs 20% chance), so grounding
failure is a **retrieval** problem — which is what makes the Phase 3 reranker the highest-value
next step.

## Working notes

`markdown/` is gitignored — local notes, not shipped docs. `LEARNINGS.md` (every wrong turn,
broke/why/fix) is the most useful; also `EVAL_RESULTS.md`, `SAFETY.md` (licence audit for data
*and* weights), `CODING_STANDARDS.md`, `CORPUS_SOURCES.md`.