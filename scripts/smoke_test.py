"""Plain-assert sanity checks for the core pipeline pieces. No pytest needed:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models import Chunk, Prior, Sample  # noqa: E402
from backend.pipeline import merge_samples_markdown  # noqa: E402
from backend.priors import parse_priors_markdown  # noqa: E402
from backend.selection import ClusterRepresentativeSelector  # noqa: E402
from backend.text_utils import split_sentences  # noqa: E402


def test_priors_parsing_splits_on_separators():
    text = "First quote.\n\n---\n\nSecond quote.\n\n-------------------\n\nThird quote."
    priors = parse_priors_markdown(text)
    assert len(priors) == 3, f"expected 3 priors, got {len(priors)}"
    assert priors[0].text == "First quote."
    assert priors[2].text == "Third quote."


def test_split_sentences_handles_abbreviations():
    sentences = split_sentences("Mr. Smith arrived. He left.")
    assert len(sentences) == 2, f"expected 2 sentences, got {sentences!r}"


def test_cosine_self_similarity_is_one():
    rng = np.random.default_rng(0)
    v = rng.normal(size=8)
    v = v / np.linalg.norm(v)
    assert abs(float(v @ v) - 1.0) < 1e-6


def _make_chunk(book_id: str, idx: int) -> Chunk:
    return Chunk(
        id=str(idx),
        book_id=book_id,
        text=f"chunk {idx}",
        start_sentence_idx=idx * 10,
        end_sentence_idx=idx * 10 + 5,
        window_label="3s",
    )


def test_cluster_selector_scans_whole_book_but_returns_only_safe_chunks():
    # Two obvious 2D clusters: {0, 1} lean +x, {2, 3} lean +y.
    chunks = [_make_chunk("b", i) for i in range(4)]
    candidate_emb = np.array(
        [
            [1.0, 0.0],    # chunk 0: safe, +x cluster
            [0.9, 0.436],  # chunk 1: UNSAFE (late in book), +x cluster
            [0.436, 0.9],  # chunk 2: UNSAFE (late in book), +y cluster
            [0.0, 1.0],    # chunk 3: safe, +y cluster
        ],
        dtype=np.float32,
    )
    position_fractions = np.array([0.05, 0.9, 0.9, 0.1])
    priors = [Prior(id="p0", text="prior zero", n_sentences=1, n_words=2)]
    prior_emb = np.array([[1.0, 0.0]], dtype=np.float32)

    selector = ClusterRepresentativeSelector()
    samples = selector.select(chunks, candidate_emb, position_fractions, priors, prior_emb, k=2, spoiler_fraction=0.2)

    returned_ids = {s.chunk.id for s in samples}
    assert returned_ids == {"0", "3"}, f"unsafe (late-book) chunks leaked into output: {returned_ids}"

    kinds = [s.kind for s in samples]
    assert kinds.count("taste") == 2, "expected one taste-match per cluster"
    assert kinds.count("representative") == 2, "expected one representative pick per cluster"

    for s in samples:
        if s.kind == "taste":
            assert s.matched_prior_text == "prior zero"
        else:
            assert s.matched_prior_text == ""


def test_merge_samples_markdown_sections_and_details():
    taste = Sample(
        chunk=_make_chunk("b", 0), score=0.5, rank=1, kind="taste",
        matched_prior_text="a prior", position_fraction=0.1,
    )
    representative = Sample(
        chunk=_make_chunk("b", 1), score=0.4, rank=1, kind="representative", position_fraction=0.05,
    )

    doc = merge_samples_markdown("My Book", [taste, representative])

    assert "## Matches Your Taste" in doc
    assert "## Representative of This Book" in doc
    assert "<details>" in doc and "<summary>Match details</summary>" in doc
    assert "a prior" in doc
    assert "~10% into the book" in doc
    # representative samples are Priors-agnostic — no "Closest Prior" line for them
    assert doc.count("Closest Prior") == 1


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} smoke tests passed")


if __name__ == "__main__":
    main()
