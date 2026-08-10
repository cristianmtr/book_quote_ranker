"""Plain-assert sanity checks for the core pipeline pieces. No pytest needed:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models import Chunk  # noqa: E402
from backend.pipeline import _truncate_to_fraction  # noqa: E402
from backend.priors import parse_priors_markdown  # noqa: E402
from backend.selection import NearestPriorMMRSelector  # noqa: E402
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


def test_mmr_with_k1_reduces_to_argmax():
    chunks = [_make_chunk("b", i) for i in range(5)]
    candidate_emb = np.eye(5, dtype=np.float32)  # mutually orthogonal, no overlap concerns
    prior_emb = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # nearest to chunk 2

    selector = NearestPriorMMRSelector()
    samples = selector.select(chunks, candidate_emb, prior_emb, k=1)

    assert len(samples) == 1
    assert samples[0].chunk.id == chunks[2].id


def test_truncate_to_fraction_keeps_only_early_paragraphs():
    paragraphs = [f"word{i} " * 10 for i in range(10)]  # 10 paragraphs, 10 words each
    text = "\n\n".join(paragraphs)

    truncated = _truncate_to_fraction(text, 0.2)

    assert truncated == "\n\n".join(paragraphs[:2]), "expected only the first ~20% of paragraphs"
    assert "word9" not in truncated, "spoiler guard leaked a late paragraph"


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} smoke tests passed")


if __name__ == "__main__":
    main()
