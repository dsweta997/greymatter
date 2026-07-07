# Gray Matter

An illustrated medical-education tutor for students — grounded, cited answers from open
medical textbooks, with each step illustrated in the style of the public-domain 1918
Gray's Anatomy plates.

> **Education, not advice.** Gray Matter is a study tool for anatomy, physiology, and
> pathophysiology *concepts*. It is **not** a medical-advice tool: it does not diagnose,
> recommend treatment or drug doses, or interpret your personal results. Personal-health
> and crisis questions are refused and redirected to real professionals.

## Status

**Phase 1 complete** — the data foundation:
- **Corpus**: 24,922 openly-licensed documents (4 OpenStax textbooks + MedQuAD) → `corpus/corpus.parquet`
- **Retrieval eval set**: 100 stratified, corpus-grounded questions → `corpus/eval/eval_set.parquet`
- **Guardrail set**: 100 labeled questions (educational / personal-advice / crisis) → `corpus/eval/guardrail_set.parquet`

**Phase 2 in progress** — dense retrieval:
- **Vector index**: all 24,922 chunks embedded with `bge-small-en-v1.5` (384-dim) and indexed into embedded Qdrant (cosine). Built in ~90s on GPU.
- **Retriever**: `rag/retriever.py` — query → top-k chunks by cosine.
- Next: baseline hit-rate on the eval set, then the generator + CLI.

## Repo layout

- `data_pipeline/` — numbered pipeline scripts, in run order: ingest → build eval set → build guardrail set → export Parquet → embed + index
- `rag/` — query-time system (retriever; generator + pipeline to come)
- `common/` — shared helpers (config loading, device, vector-store client)
- `configs/config.yaml` — all settings (sources, licenses, chunking, embedding, vector store)
