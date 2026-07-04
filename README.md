# Gray Matter

An illustrated medical-education tutor for students — grounded, cited answers from open
medical textbooks, with each step illustrated in the style of the public-domain 1918
Gray's Anatomy plates.

> ⚠️ **Education, not advice.** Gray Matter is a study tool for anatomy, physiology, and
> pathophysiology *concepts*. It is **not** a medical-advice tool: it does not diagnose,
> recommend treatment or drug doses, or interpret your personal results. Personal-health
> and crisis questions are refused and redirected to real professionals.

## Status

**Phase 1 complete** — the data foundation:
- **Corpus**: 24,922 openly-licensed documents (4 OpenStax textbooks + MedQuAD) → `corpus/corpus.parquet`
- **Retrieval eval set**: 100 stratified, corpus-grounded questions → `corpus/eval/eval_set.parquet`
- **Guardrail set**: 100 labeled questions (educational / personal-advice / crisis) → `corpus/eval/guardrail_set.parquet`

Every document carries its license and attribution. See `SAFETY.md` for the security +
licensing audit, `CORPUS_SOURCES.md` for the source map, and `LEARNINGS.md` for the build log.

## Repo layout

- `data_pipeline/` — corpus ingestion + Parquet export
- `evaluation/` — eval-set and guardrail-set builders
- `configs/config.yaml` — all settings (sources, licenses, chunking, license gate)

*Next: Phase 2 — baseline RAG (chunk → embed → index → retrieve).*
