"""
Phase 2, Feature 2 — the retriever (online half of RAG, step 1).

Wraps the dense vector index into a reusable Retriever: embed a query (WITH the
bge query prefix), search Qdrant by cosine, return the top-k chunks with payloads.
Shared by the generator, the CLI pipeline, and the retrieval eval — so the model
and the client load once and the query-side details live in exactly one place.

No orchestration framework (LangChain/LangGraph): this is single-turn retrieval,
built directly on qdrant-client + sentence-transformers for transparency.

Run (quick test):  python -m rag.retriever "what is the sinoatrial node"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer

from common import get_device, load_config, make_qdrant_client


@dataclass
class Hit:
    """One retrieved chunk. Attribute access reads cleanly in the generator/eval."""

    score: float
    doc_id: str
    title: str
    source: str
    license: str
    text: str


class Retriever:
    """Holds the embedding model + Qdrant client; turns a query into ranked Hits."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.emb = cfg["embedding"]
        self.vs = cfg["vector_store"]
        self.collection = self.vs["collection"]
        self.prefix = self.emb["query_prefix"]
        self.normalize = self.emb["normalize"]
        # ef_search is a QUERY-time HNSW knob (recall vs latency). Ignored in local
        # mode (Qdrant does exact search there) but correct once we point at a server.
        self.ef_search = self.vs.get("hnsw", {}).get("ef_search", 100)

        self.model = SentenceTransformer(self.emb["model"], device=get_device())
        self.client = make_qdrant_client(self.vs)

    # -- internals -------------------------------------------------------- #
    def _search(self, vector, k: int) -> list:
        return self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=self.ef_search),
        ).points

    @staticmethod
    def _to_hit(h) -> Hit:
        p = h.payload
        return Hit(
            score=h.score,
            doc_id=p["doc_id"],
            title=p["title"],
            source=p["source"],
            license=p["license"],
            text=p["text"],
        )

    # -- public API ------------------------------------------------------- #
    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        """Top-k chunks for one query. The bge prefix is applied to queries only."""
        vec = self.model.encode(self.prefix + query, normalize_embeddings=self.normalize)
        return [self._to_hit(h) for h in self._search(vec, k)]

    def retrieve_many(self, queries: list[str], k: int = 5) -> list[list[Hit]]:
        """Top-k for many queries; batch-encodes in one forward pass (used by eval)."""
        vecs = self.model.encode(
            [self.prefix + q for q in queries], normalize_embeddings=self.normalize
        )
        return [[self._to_hit(h) for h in self._search(v, k)] for v in vecs]

    # -- lifecycle (release the embedded-store lock on exit) -------------- #
    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main() -> int:
    query = " ".join(sys.argv[1:]) or "what is the sinoatrial node"
    with Retriever() as r:
        hits = r.retrieve(query, k=5)
    print(f"query: {query!r}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.score:.3f}  [{h.source}] {h.title[:55]}")
        print(f"   {h.text[:160].strip()}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
