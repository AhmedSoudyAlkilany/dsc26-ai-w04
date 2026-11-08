# Retrieval evaluation

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | q12 |
| Hybrid (RRF) | 0.9167 (11/12) | q12 |

The missed query (q12) searches for governing law and jurisdiction within a long bilingual agency contract (`agency_bi_08`) where general contractual terms appear repeatedly across multiple earlier clauses, pushing the specific dispute-resolution section outside the top 3. Because `HashingEmbedder` relies solely on character 3-gram hashing rather than semantic comprehension, it cannot bridge bilingual cross-lingual phrasing or distinguish clause specificity. To improve retrieval on such queries, next steps include replacing the hashing embedder with a multilingual dense embedding model (e.g. `multilingual-e5-small`), prefixing chunk headers with section titles, and incorporating a cross-encoder re-ranker.
