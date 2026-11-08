"""Reciprocal rank fusion and the hybrid retriever. You write this file."""
from __future__ import annotations

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_idx, chunk_id in enumerate(ranking):
            rank = rank_idx + 1  # ranks start at 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    results = list(scores.items())
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None):
        self._chunks = {c.chunk_id: c for c in chunks}
        self._chunk_list = chunks
        self._embedder = embedder
        self._reranker = reranker

        # Build BM25 index
        self._bm25 = BM25Index(k1=k1, b=b)
        self._bm25.add(chunks)

        # Pre-compute embeddings for all chunks
        if chunks:
            texts = [c.text for c in chunks]
            self._chunk_ids = [c.chunk_id for c in chunks]
            self._embeddings = embedder.embed(texts)  # shape (N, dim)
        else:
            self._chunk_ids = []
            self._embeddings = np.zeros((0, 0))

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        if not query or not query.strip():
            return []

        # BM25 retrieval: top `candidates`
        bm25_results = self._bm25.search(query, top_k=candidates)
        bm25_ids = [cid for cid, _ in bm25_results]

        # Dense retrieval: top `candidates` by dot product
        query_emb = self._embedder.embed([query])  # shape (1, dim)
        if self._embeddings.shape[0] > 0 and query_emb.shape[0] > 0:
            # Dot product (since embeddings are L2-normalised, this is cosine sim)
            scores = self._embeddings @ query_emb.T  # shape (N, 1)
            scores = scores.flatten()  # shape (N,)

            # Create (score, chunk_id) pairs and sort by score desc, then chunk_id asc
            dense_pairs = list(zip(scores, self._chunk_ids))
            dense_pairs.sort(key=lambda x: (-x[0], x[1]))
            dense_ids = [cid for _, cid in dense_pairs[:candidates]]
        else:
            dense_ids = []

        # Fuse with RRF
        fused = rrf([bm25_ids, dense_ids])
        fused = fused[:candidates]

        # Build Hits
        hits: list[Hit] = []
        for chunk_id, score in fused:
            chunk = self._chunks[chunk_id]
            hits.append(Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                score=score,
            ))

        # If reranker is given, call it once and re-sort
        if self._reranker is not None:
            reranker_scores = self._reranker(query, hits)
            # Replace scores and re-sort
            scored_hits = list(zip(reranker_scores, hits))
            scored_hits.sort(key=lambda x: (-x[0], x[1].chunk_id))
            hits = [
                Hit(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    start=h.start,
                    end=h.end,
                    text=h.text,
                    score=s,
                )
                for s, h in scored_hits
            ]

        return hits[:top_k]
