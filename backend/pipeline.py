from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Callable

from .chunking import ParagraphSentenceChunker
from .config import SPOILER_GUARD_FRACTION
from .embeddings import get_embedder
from .epub_io import epub_to_markdown, markdown_to_epub
from .models import BookResult, BookStatus, Sample
from .priors import parse_priors_file
from .selection import NearestPriorMMRSelector

StatusCallback = Callable[[str, BookStatus], None] | None


def _window_sizes(priors) -> list[int]:
    base = max(2, round(statistics.median(p.n_sentences for p in priors))) if priors else 3
    return sorted({max(1, round(0.5 * base)), base, round(1.5 * base)})


def _truncate_to_fraction(markdown_text: str, fraction: float) -> str:
    """Keep only the first `fraction` of the book's words (cut at paragraph
    boundaries), so later chunking/selection never sees later plot points."""
    paragraphs = markdown_text.split("\n\n")
    target_words = max(1, int(sum(len(p.split()) for p in paragraphs) * fraction))

    kept: list[str] = []
    word_count = 0
    for para in paragraphs:
        kept.append(para)
        word_count += len(para.split())
        if word_count >= target_words:
            break
    return "\n\n".join(kept)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "book"


def merge_samples_markdown(samples: list[Sample]) -> str:
    return "\n\n---\n\n".join(s.chunk.text for s in samples)


def process_book(
    epub_path: Path,
    book_id: str,
    filename: str,
    prior_emb,
    window_sizes: list[int],
    k: int,
    embedder,
    chunker,
    selector,
    out_dir: Path,
    on_status: StatusCallback = None,
) -> BookResult:
    result = BookResult(book_id=book_id, filename=filename)

    def status(s: BookStatus) -> None:
        result.status = s
        if on_status:
            on_status(book_id, s)

    try:
        status("converting")
        markdown_text = epub_to_markdown(epub_path)
        markdown_text = _truncate_to_fraction(markdown_text, SPOILER_GUARD_FRACTION)

        status("chunking")
        chunks = chunker.chunk(book_id, markdown_text, window_sizes)
        if not chunks:
            raise ValueError("no chunkable text extracted from EPUB")

        status("embedding")
        chunk_emb = embedder.embed([c.text for c in chunks])

        status("selecting")
        samples = selector.select(chunks, chunk_emb, prior_emb, k)
        result.samples = samples

        status("assembling")
        title = f"{Path(filename).stem} — Samples"
        merged_md = merge_samples_markdown(samples)
        out_path = out_dir / f"{_slugify(Path(filename).stem)}_samples.epub"
        markdown_to_epub(title, merged_md, out_path)
        result.output_epub_path = out_path

        status("done")
    except Exception as exc:  # noqa: BLE001 - isolate per-book failures so one bad EPUB doesn't fail the job
        result.error = str(exc)
        status("error")

    return result


def process_books(
    candidate_paths: list[tuple[str, Path, str]],  # (book_id, epub_path, original_filename)
    priors_path: Path,
    k: int,
    out_dir: Path,
    on_status: StatusCallback = None,
    embedder=None,
) -> list[BookResult]:
    """End-to-end pipeline: parse Priors once, derive window sizes and embed
    Priors once, then chunk/embed/select/assemble per candidate book. Single
    orchestration entry point shared by the CLI and the FastAPI job runner.

    `embedder` can be passed in as a shared, already-loaded instance (the
    FastAPI app loads it once at startup) to avoid reloading model weights
    on every job; the CLI runs once per process so it's fine to omit it.
    """
    priors = parse_priors_file(priors_path)
    if not priors:
        raise ValueError("Priors file contains no quotes")

    window_sizes = _window_sizes(priors)

    embedder = embedder or get_embedder()
    prior_emb = embedder.embed([p.text for p in priors])

    chunker = ParagraphSentenceChunker()
    selector = NearestPriorMMRSelector()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return [
        process_book(
            Path(path), book_id, filename, prior_emb, window_sizes, k,
            embedder, chunker, selector, out_dir, on_status,
        )
        for book_id, path, filename in candidate_paths
    ]
