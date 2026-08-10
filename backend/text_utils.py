"""Sentence and paragraph splitting used by priors parsing and chunking."""
from __future__ import annotations

import re

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s")


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]


def split_paragraphs(markdown_text: str) -> list[str]:
    """Split converted-book markdown into paragraphs, dropping headers and short noise
    (front matter, TOC entries, chapter numbers)."""
    raw_paragraphs = re.split(r"\n\s*\n", markdown_text)
    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        if not p or _HEADER_RE.match(p) or len(p.split()) < 3:
            continue
        paragraphs.append(p)
    return paragraphs
