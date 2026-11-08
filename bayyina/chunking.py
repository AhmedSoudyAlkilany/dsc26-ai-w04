"""Offset-preserving chunker. You write this file. See README.md for the seven rules."""
from __future__ import annotations

from bayyina.types import Chunk


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text.

    ValueError unless max_chars > 0 and 0 <= overlap < max_chars.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError(
            f"Invalid arguments: max_chars={max_chars}, overlap={overlap}. "
            f"Need max_chars > 0 and 0 <= overlap < max_chars."
        )

    if not text or text.isspace():
        return []

    chunks: list[Chunk] = []
    idx = 0
    s = 0

    # Skip initial leading whitespace
    while s < len(text) and text[s].isspace():
        s += 1

    while s < len(text):
        if len(text) - s <= max_chars:
            chunk_str = text[s:].rstrip()
            if chunk_str:
                end = s + len(chunk_str)
                chunks.append(Chunk(doc_id, f"{doc_id}#{idx}", s, end, chunk_str))
                idx += 1
            break

        window = text[s : s + max_chars]
        half = max_chars // 2

        # Priority 1: paragraph break \n\n in second half of window
        c = None
        for ci in range(max_chars, half - 1, -1):
            if ci >= 2 and window[ci - 2 : ci] == "\n\n":
                c = ci
                is_mid_word_cut = False
                break

        # Priority 2: sentence end in second half of window
        if c is None:
            sentence_ends = {'.', '!', '?', '؟', '۔', '\n'}
            for ci in range(max_chars, half - 1, -1):
                if window[ci - 1] in sentence_ends:
                    c = ci
                    is_mid_word_cut = False
                    break

        # Priority 3: space in second half of window
        if c is None:
            for ci in range(max_chars, half - 1, -1):
                if window[ci - 1].isspace():
                    c = ci
                    is_mid_word_cut = False
                    break

        # Priority 4: hard cut at window end (cut mid-word)
        if c is None:
            c = max_chars
            raw_end = s + c
            is_mid_word_cut = (
                raw_end < len(text)
                and raw_end > 0
                and not text[raw_end - 1].isspace()
                and not text[raw_end].isspace()
            )

        raw_end = s + c
        chunk_str = text[s:raw_end].rstrip()
        end = s + len(chunk_str)

        chunks.append(Chunk(doc_id, f"{doc_id}#{idx}", s, end, chunk_str))
        idx += 1

        next_s = max(s + 1, end - overlap)

        if not is_mid_word_cut:
            # Must start at the beginning of a word
            if next_s > 0 and next_s < len(text) and not text[next_s - 1].isspace() and not text[next_s].isspace():
                while next_s < len(text) and not text[next_s].isspace():
                    next_s += 1
            while next_s < len(text) and text[next_s].isspace():
                next_s += 1
        else:
            # May start mid-word, but cannot start with whitespace
            while next_s < len(text) and text[next_s].isspace():
                next_s += 1

        s = next_s

    return chunks
