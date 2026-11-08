"""BM25, written by hand. You write this file (no search libraries)."""
from __future__ import annotations

import math
from collections import Counter

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunk_ids: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter] = []  # term frequencies per document
        self._doc_lens: list[int] = []
        self._df: Counter = Counter()  # document frequency for each term
        self._N: int = 0
        self._avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self._chunk_ids.append(chunk.chunk_id)
            self._doc_tokens.append(tokens)
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(tokens))

            # Update document frequency: each unique term in this chunk
            for term in tf:
                self._df[term] += 1

        self._N = len(self._chunk_ids)
        total_len = sum(self._doc_lens)
        self._avgdl = total_len / self._N if self._N > 0 else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""
        if self._N == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Each distinct query term counts once
        query_terms = set(query_tokens)

        # Precompute IDF for query terms
        idf: dict[str, float] = {}
        for term in query_terms:
            df = self._df.get(term, 0)
            idf[term] = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

        results: list[tuple[str, float]] = []
        k1 = self.k1
        b = self.b
        avgdl = self._avgdl

        for i in range(self._N):
            chunk_id = self._chunk_ids[i]
            tf_map = self._doc_freqs[i]
            dl = self._doc_lens[i]

            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                term_idf = idf[term]
                numerator = tf * (k1 + 1)
                norm_dl = (dl / avgdl) if avgdl > 0 else 0.0
                denominator = tf + k1 * (1 - b + b * norm_dl)
                score += term_idf * numerator / denominator

            if score > 0:
                results.append((chunk_id, score))

        # Sort by score descending, then chunk ID ascending
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]
