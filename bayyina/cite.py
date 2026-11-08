"""Citation checks. You write this file."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    """True only when 0 <= start < end <= len(document), the quote is not blank,
    and the quote (whitespace collapsed) occurs inside document[start:end]
    (whitespace collapsed)."""
    # Boundary checks
    if not (0 <= start < end <= len(document)):
        return False

    # Quote must not be blank
    if not quote or not quote.strip():
        return False

    # Whitespace-collapsed comparison
    collapsed_quote = " ".join(quote.split())
    collapsed_span = " ".join(document[start:end].split())

    return collapsed_quote in collapsed_span


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """[doc_id:start-end] "quote"

    Where the quote is the hit's text with whitespace collapsed; when it is
    longer than max_quote, take its first max_quote characters, cut back to
    the last space inside them (dropping the space), and append …
    """
    quote = " ".join(hit.text.split())

    if len(quote) > max_quote:
        truncated = quote[:max_quote]
        last_space = truncated.rfind(' ')
        if last_space != -1:
            truncated = truncated[:last_space]
        quote = truncated + '…'

    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
