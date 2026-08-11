from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import ParagraphSentenceChunker
from .config import SPOILER_GUARD_FRACTION
from .embeddings import get_embedder
from .epub_io import epub_to_markdown
from .models import BookResult, BookStatus, Prior, Sample
from .priors import parse_priors_file
from .selection import ClusterRepresentativeSelector
from .text_utils import split_paragraphs, split_sentences

StatusCallback = Callable[[str, BookStatus], None] | None

SECTION_TITLES = {"taste": "Matches Your Taste", "representative": "Representative of This Book"}


def _window_sizes(priors: list[Prior]) -> list[int]:
    base = max(2, round(statistics.median(p.n_sentences for p in priors))) if priors else 3
    return sorted({max(1, round(0.5 * base)), base, round(1.5 * base)})


def _total_sentence_count(markdown_text: str) -> int:
    return sum(len(split_sentences(p)) for p in split_paragraphs(markdown_text))


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "book"


def _sample_block(s: Sample) -> str:
    position_pct = round(s.position_fraction * 100)
    lines = [
        "<details>",
        "<summary>Match details</summary>",
        "",
        f"- **Rank:** {s.rank}",
        f"- **Score:** {s.score:.3f}",
        f"- **Position:** ~{position_pct}% into the book",
        f"- **Window size:** {s.chunk.window_label}",
    ]
    if s.kind == "taste":
        lines.append(f"- **Closest Prior:** {s.matched_prior_text}")
    lines += ["", "</details>"]
    return f"{s.chunk.text}\n\n" + "\n".join(lines)


def merge_samples_markdown(title: str, samples: list[Sample]) -> str:
    """Assembles the downloadable Samples document: a "Matches Your Taste"
    section followed by a "Representative of This Book" section, each excerpt
    followed by a collapsible <details> block with its match metadata, all
    separated by horizontal rules."""
    blocks = [f"# {title}"]
    for kind in ("taste", "representative"):
        group = [s for s in samples if s.kind == kind]
        if not group:
            continue
        blocks.append(f"## {SECTION_TITLES[kind]}")
        blocks.extend(_sample_block(s) for s in group)
    return "\n\n---\n\n".join(blocks)


def process_book(
    epub_path: Path,
    book_id: str,
    filename: str,
    priors: list[Prior],
    prior_emb,
    window_sizes: list[int],
    k: int,
    spoiler_fraction: float,
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
        markdown_text = epub_to_markdown(epub_path)  # full book — no truncation
        total_sentences = _total_sentence_count(markdown_text)

        status("chunking")
        chunks = chunker.chunk(book_id, markdown_text, window_sizes)
        if not chunks:
            raise ValueError("no chunkable text extracted from EPUB")

        status("embedding")
        chunk_emb = embedder.embed([c.text for c in chunks])
        position_fractions = np.array(
            [c.start_sentence_idx / total_sentences if total_sentences else 0.0 for c in chunks]
        )

        status("clustering")
        samples = selector.select(
            chunks, chunk_emb, position_fractions, priors, prior_emb, k, spoiler_fraction
        )
        result.samples = samples

        taste_samples = [s for s in samples if s.kind == "taste"]
        result.aggregate_score = (
            sum(s.score for s in taste_samples) / len(taste_samples) if taste_samples else 0.0
        )

        status("assembling")
        title = f"{Path(filename).stem} — Samples"
        merged_md = merge_samples_markdown(title, samples)
        out_path = out_dir / f"{_slugify(Path(filename).stem)}_samples.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(merged_md, encoding="utf-8")
        result.output_path = out_path

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
    spoiler_fraction: float = SPOILER_GUARD_FRACTION,
) -> list[BookResult]:
    """End-to-end pipeline: parse Priors once, derive window sizes and embed
    Priors once, then chunk/embed/cluster/assemble per candidate book. Single
    orchestration entry point shared by the CLI and the FastAPI job runner.

    Every candidate book is chunked and embedded in full (no truncation) so
    that clustering reflects the book's real structure; `spoiler_fraction`
    is applied inside the selector, restricting only which chunks may be
    *returned*, not what's scanned to determine representativeness.

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
    selector = ClusterRepresentativeSelector()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return [
        process_book(
            Path(path), book_id, filename, priors, prior_emb, window_sizes, k, spoiler_fraction,
            embedder, chunker, selector, out_dir, on_status,
        )
        for book_id, path, filename in candidate_paths
    ]
