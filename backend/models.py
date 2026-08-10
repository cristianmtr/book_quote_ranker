from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

BookStatus = Literal[
    "pending", "converting", "chunking", "embedding", "selecting", "assembling", "done", "error"
]


@dataclass
class Prior:
    id: str
    text: str
    n_sentences: int
    n_words: int


@dataclass
class Chunk:
    id: str
    book_id: str
    text: str
    start_sentence_idx: int
    end_sentence_idx: int
    window_label: str


@dataclass
class Sample:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class BookResult:
    book_id: str
    filename: str
    status: BookStatus = "pending"
    samples: list[Sample] = field(default_factory=list)
    output_epub_path: Path | None = None
    error: str | None = None


# --- pydantic models: cross the HTTP boundary ---


class BookResultSummary(BaseModel):
    book_id: str
    filename: str
    status: str
    sample_count: int
    error: str | None = None


class JobStatus(BaseModel):
    job_id: str
    state: Literal["uploaded", "processing", "done", "error"]
    k: int
    books: list[BookResultSummary]
    created_at: datetime
    updated_at: datetime
    error: str | None = None
