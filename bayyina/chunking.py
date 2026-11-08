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

    # Rule 7: empty or whitespace-only text returns []
    if not text or text.isspace():
        return []

    chunks: list[Chunk] = []
    idx = 0
    s = 0

    # Skip initial leading whitespace so chunk 0 starts at a non-whitespace character
    while s < len(text) and text[s].isspace():
        s += 1

    while s < len(text):
        remaining = len(text) - s
        if remaining <= max_chars:
            # Reaches the end of the text (minus trailing whitespace)
            chunk_stripped = text[s:].rstrip()
            if chunk_stripped:
                end = s + len(chunk_stripped)
                chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{idx}",
                    start=s,
                    end=end,
                    text=text[s:end],
                ))
                idx += 1
            break

        window = text[s : s + max_chars]
        half = max_chars // 2

        # Priority 1: paragraph break \n\n (positions c-2, c-1 in window)
        best_c = -1
        for c in range(max_chars, half - 1, -1):
            if c >= 2 and window[c - 2] == '\n' and window[c - 1] == '\n':
                best_c = c
                break

        # Priority 2: sentence end (. ! ? ؟ ۔ or \n at position c-1)
        if best_c == -1:
            sentence_ends = {'.', '!', '?', '\u061F', '\u06D4', '\n'}
            for c in range(max_chars, half - 1, -1):
                if c >= 1 and window[c - 1] in sentence_ends:
                    best_c = c
                    break

        # Priority 3: space (any whitespace character at position c-1)
        if best_c == -1:
            for c in range(max_chars, half - 1, -1):
                if c >= 1 and window[c - 1].isspace():
                    best_c = c
                    break

        # Priority 4: hard cut at max_chars
        if best_c == -1:
            best_c = max_chars

        raw_end = s + best_c
        chunk_raw = text[s:raw_end]
        chunk_stripped = chunk_raw.rstrip()

        if not chunk_stripped:
            # Advance past whitespace
            s = raw_end
            while s < len(text) and text[s].isspace():
                s += 1
            continue

        actual_end = s + len(chunk_stripped)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}#{idx}",
            start=s,
            end=actual_end,
            text=text[s:actual_end],
        ))
        idx += 1

        # A chunk ended with a mid-word cut when the characters on both sides
        # of its end are not whitespace.
        prev_ended_mid_word = (
            actual_end < len(text)
            and actual_end > 0
            and not text[actual_end - 1].isspace()
            and not text[actual_end].isspace()
        )

        if prev_ended_mid_word:
            # May start mid-word
            next_s = max(s + 1, actual_end - overlap)
            while next_s < len(text) and text[next_s].isspace():
                next_s += 1
        else:
            # Must start at the beginning of a word
            # (e.g. the first word inside the overlap, or the first word after the cut)
            min_start = max(s + 1, actual_end - overlap)
            found_word = None
            for p in range(min_start, actual_end + 1):
                if p < len(text) and not text[p].isspace() and (p == 0 or text[p - 1].isspace()):
                    found_word = p
                    break
            if found_word is not None:
                next_s = found_word
            else:
                p = actual_end
                while p < len(text) and text[p].isspace():
                    p += 1
                next_s = p

        s = next_s

    return chunks
