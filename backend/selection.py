from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import MMR_LAMBDA, pool_size
from .models import Chunk, Sample


class Selector(Protocol):
    def select(
        self,
        candidates: list[Chunk],
        candidate_emb: np.ndarray,
        prior_emb: np.ndarray,
        k: int,
    ) -> list[Sample]: ...


class Reranker(Protocol):
    """Future drop-in point: re-score a shortlist of embedding-selected
    Samples (e.g. with an LLM-as-judge prompt against the Priors) to catch
    style/wit/imagery nuance embeddings miss. Not implemented in v1 —
    NoOpReranker is the default and pipeline.py does not call this yet."""

    def rerank(self, samples: list[Sample], priors_text: list[str]) -> list[Sample]: ...


class NoOpReranker:
    def rerank(self, samples: list[Sample], priors_text: list[str]) -> list[Sample]:
        return samples


def _overlap_frac(a: Chunk, b: Chunk) -> float:
    lo = max(a.start_sentence_idx, b.start_sentence_idx)
    hi = min(a.end_sentence_idx, b.end_sentence_idx)
    overlap = max(0, hi - lo + 1)
    a_len = a.end_sentence_idx - a.start_sentence_idx + 1
    return overlap / a_len if a_len else 0.0


class NearestPriorMMRSelector:
    """Relevance = max cosine similarity to any single Prior (not centroid),
    since Priors are thematically diverse. Diversity via MMR over a bounded
    top-relevance pool, with a hard skip on heavily-overlapping sentence
    ranges (guards against near-duplicate windows of different sizes)."""

    def __init__(self, mmr_lambda: float = MMR_LAMBDA, pool_size_fn=pool_size):
        self.mmr_lambda = mmr_lambda
        self.pool_size_fn = pool_size_fn

    def select(
        self,
        candidates: list[Chunk],
        candidate_emb: np.ndarray,
        prior_emb: np.ndarray,
        k: int,
    ) -> list[Sample]:
        if not candidates:
            return []

        relevance = (candidate_emb @ prior_emb.T).max(axis=1)

        pool_n = min(len(candidates), self.pool_size_fn(k))
        pool_idx = list(np.argsort(-relevance)[:pool_n])

        selected: list[int] = []
        remaining = list(pool_idx)

        while len(selected) < k and remaining:
            if not selected:
                pick = max(remaining, key=lambda i: relevance[i])
            else:

                def mmr_score(i: int) -> float:
                    if any(_overlap_frac(candidates[i], candidates[j]) > 0.5 for j in selected):
                        return float("-inf")
                    diversity_penalty = max(float(candidate_emb[i] @ candidate_emb[j]) for j in selected)
                    return self.mmr_lambda * relevance[i] - (1 - self.mmr_lambda) * diversity_penalty

                scored = [(mmr_score(i), i) for i in remaining]
                best_score, pick = max(scored, key=lambda t: t[0])
                if best_score == float("-inf"):
                    break  # everything left overlaps too heavily with what's already picked
            selected.append(pick)
            remaining.remove(pick)

        samples = [
            Sample(chunk=candidates[i], score=float(relevance[i]), rank=0) for i in selected
        ]
        samples.sort(key=lambda s: s.score, reverse=True)
        for rank, s in enumerate(samples, start=1):
            s.rank = rank
        return samples
