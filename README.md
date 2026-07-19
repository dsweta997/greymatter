# Gray Matter

An illustrated medical-education tutor for students — grounded, cited answers from open
medical textbooks, with each step illustrated in the style of the public-domain 1918
Gray's Anatomy plates.

> **Education, not advice.** Gray Matter is a study tool for anatomy, physiology, and
> pathophysiology *concepts*. It is **not** a medical-advice tool: it must not be used to
> diagnose, recommend treatment or drug doses, or interpret your personal results.
>
> **Not yet safe to use as a tutor.** Refusing personal-health questions and redirecting
> crisis questions is the *design intent*, enforced today only by the generator's system
> prompt — and measurement shows that is not sufficient. On the current build, personal-advice
> questions are refused **26%** of the time (target ≥95%) and crisis questions are redirected
> to help **2 times in 8** (target 8/8). The input classifier that enforces this properly is
> Phase 3.5 and is **not built yet**. Until it lands, this is a development artifact.

## Status

**Phase 1 complete** — the data foundation:
- **Corpus**: 24,922 openly-licensed documents (4 OpenStax textbooks + MedQuAD) → `corpus/corpus.parquet`
- **Retrieval eval set**: 100 stratified, corpus-grounded questions → `corpus/eval/eval_set.parquet`
- **Guardrail set**: 100 labeled questions (educational / personal-advice / crisis) → `corpus/eval/guardrail_set.parquet`

**Phase 2 complete** — baseline RAG, end to end:
- **Vector index**: all 24,922 chunks embedded with `bge-small-en-v1.5` (384-dim) and indexed into embedded Qdrant (cosine). Built in ~90s on GPU.
- **Retriever**: `rag/retriever.py` — query → top-k chunks by cosine.
- **Generator**: `rag/generator.py` — grounded, cited answers via any OpenAI-compatible endpoint (Ollama locally; `qwen2.5:7b`, Apache 2.0).
- **Pipeline CLI**: `rag/pipeline.py` — prints retrieved context *and* answer, so failures are attributable to retrieval vs generation.
- **Measured**: retrieval hit@5 = **0.656** (target ≥0.80). Citation coverage is **0.76** — but
  it scores the same whether or not retrieval found the right document, so it is a *formatting*
  metric, not evidence of grounding. Faithfulness (criterion 2) is not yet measured.

**Phase 3 next** — hybrid retrieval (BM25 + RRF) and cross-encoder reranking to close the
hit@5 gap, then the Phase 3.5 safety classifier. Both baselines are already recorded, so
each change is measured against a real "before".

```bash
python -m rag.pipeline "why do we have two circulatory loops"   # ask a question
python evaluation/run_retrieval_eval.py                          # hit@k + MRR
python evaluation/run_generation_eval.py --all                   # citations + guardrails
```

## Repo layout

- `data_pipeline/` — numbered pipeline scripts, in run order: ingest → build eval set → build guardrail set → export Parquet → embed + index
- `rag/` — query-time system: retriever, generator, pipeline CLI
- `evaluation/` — measurement scripts (retrieval and generation), judge-free
- `common/` — shared helpers (config loading, device, vector-store client)
- `configs/config.yaml` — all settings (sources, licenses, chunking, embedding, vector store, LLM)
