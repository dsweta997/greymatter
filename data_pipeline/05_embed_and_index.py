"""
Phase 2, Feature 1 — build the dense vector index.

Reads corpus/corpus.parquet, embeds each chunk with a bi-encoder (bge-small),
and upserts (vector + payload) into an embedded Qdrant collection. This is the
one-time OFFLINE half of RAG; retrieval (the online half) reads this index.

We index the corpus AS-IS (no re-chunking): OpenStax rows are already sentence-aware
~1,000-char section chunks and MedQuAD rows are self-contained Q&A. A blind character
splitter here would only destroy those boundaries.

Run:  python data_pipeline/05_embed_and_index.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import shared pkgs

import pandas as pd  # noqa: E402  (after path bootstrap)
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import get_device, load_config, make_qdrant_client  # noqa: E402

CORPUS = Path("corpus/corpus.parquet")


def main() -> int:
    cfg = load_config()
    emb, vs = cfg["embedding"], cfg["vector_store"]

    # 1. Load the corpus (documents to embed).
    df = pd.read_parquet(CORPUS)
    texts = df["text"].tolist()
    print(f"loaded {len(df)} chunks from {CORPUS}")

    # 2. Load the bi-encoder. It maps each text -> one dense vector.
    device = get_device()
    print(f"using device: {device}")
    model = SentenceTransformer(emb["model"], device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"embedding model: {emb['model']}  ->  {dim} dims")

    # 3. Embed all documents in batches. NOTE: no query prefix on documents —
    #    the bge prefix is query-side only (asymmetric search).
    vectors = model.encode(
        texts,
        batch_size=emb["batch_size"],
        normalize_embeddings=emb["normalize"],  # L2-normalize -> cosine == dot product
        show_progress_bar=True,
    )

    # 4. (Re)create the Qdrant collection sized to the model's output dim.
    client = make_qdrant_client(vs)
    if client.collection_exists(vs["collection"]):
        client.delete_collection(vs["collection"])

    # HNSW build-time tuning: m=16 neighbors, ef_construct=200 (index quality).
    # NOTE: ef_search (query breadth) is NOT set here — it's a per-query parameter
    # applied at retrieval time via SearchParams(hnsw_ef=...). Also note local mode
    # does exact search and ignores HNSW; these settings take effect on a real server.
    hnsw_cfg = vs.get("hnsw", {})
    client.create_collection(
        collection_name=vs["collection"],
        vectors_config=VectorParams(size=dim, distance=Distance[vs["distance"].upper()]),
        hnsw_config=HnswConfigDiff(
            m=hnsw_cfg.get("m", 16),
            ef_construct=hnsw_cfg.get("ef_construct", 200),
        ),
    )

    # 5. Build points. Our doc id is an md5 hex (128-bit) -> a valid UUID point id;
    #    the original id + full payload ride along so retrieval returns the chunk.
    ids = df["id"].tolist()
    points = [
        PointStruct(
            id=str(uuid.UUID(hex=ids[i])),
            vector=vectors[i].tolist(),
            payload={
                "doc_id": ids[i],
                "text": df["text"].iloc[i],
                "title": df["title"].iloc[i],
                "source": df["source"].iloc[i],
                "license": df["license"].iloc[i],
                "metadata": df["metadata"].iloc[i],
            },
        )
        for i in range(len(df))
    ]

    # 6. Upsert in batches (avoids one giant request).
    B = 256
    for s in tqdm(range(0, len(points), B), desc=f"[{vs['collection']}] upserting"):
        client.upsert(collection_name=vs["collection"], points=points[s : s + B])
    n = client.count(vs["collection"]).count
    print(f"indexed {n} vectors into collection '{vs['collection']}' at {vs['path']}/")

    # 7. Smoke test: embed test queries (WITH prefix) and verify ranking.
    test_queries = [
        "why does the left ventricle have a thicker wall",
        "what is myocardial infarction",
        "how does the citric acid cycle produce ATP",
    ]
    print("\nsmoke-test queries (top-5 per query):")
    for q in test_queries:
        qv = model.encode(emb["query_prefix"] + q, normalize_embeddings=emb["normalize"])
        hits = client.query_points(vs["collection"], query=qv.tolist(), limit=5).points
        print(f"\n  Q: {q!r}")
        for i, h in enumerate(hits, 1):
            print(f"    {i}. {h.score:.3f}  [{h.payload['source']}] {h.payload['title'][:50]}")

    client.close()  # release the on-disk lock so the retriever can open the store next
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
