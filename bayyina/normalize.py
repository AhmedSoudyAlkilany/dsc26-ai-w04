"""Arabic-aware normalisation and tokenisation. You write this file."""
from __future__ import annotations

import re

# Diacritics to remove: U+064B–U+0652 and U+0670
_DIACRITICS = set(chr(c) for c in range(0x064B, 0x0653)) | {"\u0670"}

# Tatweel U+0640
_TATWEEL = "\u0640"

# Map أ إ آ ٱ → ا
_ALEF_MAP = {
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0622": "\u0627",
    "\u0671": "\u0627",
}

# ى → ي
_ALEF_MAQSURA = "\u0649"
_YA = "\u064A"

# ة → ه
_TA_MARBUTA = "\u0629"
_HA = "\u0647"

# ٠–٩ and ۰–۹ to 0–9
_DIGIT_MAP = {}
for i in range(10):
    _DIGIT_MAP[chr(0x0660 + i)] = chr(ord("0") + i)
    _DIGIT_MAP[chr(0x06F0 + i)] = chr(ord("0") + i)

_TRANS = {}
for ch in _DIACRITICS:
    _TRANS[ord(ch)] = None
_TRANS[ord(_TATWEEL)] = None
for src, dst in _ALEF_MAP.items():
    _TRANS[ord(src)] = dst
_TRANS[ord(_ALEF_MAQSURA)] = _YA
_TRANS[ord(_TA_MARBUTA)] = _HA
for src, dst in _DIGIT_MAP.items():
    _TRANS[ord(src)] = dst

_TRANSLATE_TABLE = str.maketrans(_TRANS)
_WS_RUN = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize_ar(text: str) -> str:
    """Remove U+064B-U+0652 and U+0670; remove tatweel U+0640; map أ إ آ ٱ -> ا, ى -> ي, ة -> ه;
    map ٠-٩ and ۰-۹ to 0-9; lowercase; collapse runs of whitespace to one space and strip."""
    text = text.translate(_TRANSLATE_TABLE)
    text = text.lower()
    text = _WS_RUN.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """normalize_ar, then every match of the regex \\w+ (Unicode)."""
    return _WORD.findall(normalize_ar(text))
