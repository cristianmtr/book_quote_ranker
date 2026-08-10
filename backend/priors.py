from __future__ import annotations

import re
import uuid
from pathlib import Path

from .models import Prior
from .text_utils import split_sentences

_SEPARATOR_RE = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)


def parse_priors_markdown(text: str) -> list[Prior]:
    """Split a Priors markdown file on horizontal rules into individual quotes.

    Separators in the wild vary (`---`, `-------------------`, ...), hence `-{3,}`.
    Multi-paragraph quotes use single newlines between paragraphs (not blank lines),
    so lines are joined with spaces to normalize before sentence splitting.
    """
    priors = []
    for part in _SEPARATOR_RE.split(text):
        cleaned = " ".join(line.strip() for line in part.strip().splitlines() if line.strip())
        if not cleaned:
            continue
        priors.append(
            Prior(
                id=str(uuid.uuid4()),
                text=cleaned,
                n_sentences=len(split_sentences(cleaned)),
                n_words=len(cleaned.split()),
            )
        )
    return priors


def parse_priors_file(path: Path) -> list[Prior]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_priors_markdown(text)
