# AI usage

## Tools I used, and for what

Claude Opus and Gemini for drafting initial implementations and tests for `normalize.py`, `chunking.py`, `bm25.py`, `hybrid.py`, and `cite.py`.

## Where AI-written chunking or BM25 code broke one of the spec's rules

In `chunking.py`, the AI-generated code initially failed Rule 1 and Rule 6: when advancing to the next chunk after a non-mid-word cut (such as after punctuation or whitespace), the overlap window could land on whitespace or in the middle of a word. Because it only checked whether the previous cut was mid-word without ensuring the next chunk starts at the beginning of a word, it produced chunks with leading whitespace and mid-word starts following normal punctuation cuts. This was caught through property testing with randomized string sequences, and was fixed by explicitly seeking the first word boundary inside the overlap window or right after the cut.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | q12 |
| Hybrid (RRF) | 0.9167 (11/12) | q12 |

Hybrid tied with BM25 and dense retrieval at 11/12 recall@3 because `HashingEmbedder` acts on surface-level character 3-gram hashes rather than semantic representations, producing candidate rankings largely overlapping with BM25 keyword matches, which was insufficient to resolve the bilingual jurisdiction query q12.
