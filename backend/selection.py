from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.cluster import KMeans

from .models import Chunk, Prior, Sample


class Selector(Protocol):
    def select(
        self,
        candidates: list[Chunk],
        candidate_emb: np.ndarray,
        position_fractions: np.ndarray,
        priors: list[Prior],
        prior_emb: np.ndarray,
        k: int,
        spoiler_fraction: float,
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


class ClusterRepresentativeSelector:
    """K-means clusters chunk embeddings across the WHOLE book, so cluster
    structure reflects the book's actual thematic/stylistic makeup — not
    just its early pages. Each of the K clusters stands in for one of the
    book's major modes/themes.

    Output is deliberately two parallel groups, both restricted to chunks
    positioned within the spoiler-safe window (spoiler_fraction):
      - "taste":          per cluster, the chunk most similar to any single
                           Prior (best taste-match example of that theme).
      - "representative": per cluster, the chunk closest to that cluster's
                           centroid (typifies that theme, Priors-agnostic).

    Scanning happens over every chunk in the book; only the *selection* is
    restricted to the safe window, so representativeness is judged against
    the book's real structure while nothing past the spoiler boundary is
    ever returned.
    """

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def select(
        self,
        candidates: list[Chunk],
        candidate_emb: np.ndarray,
        position_fractions: np.ndarray,
        priors: list[Prior],
        prior_emb: np.ndarray,
        k: int,
        spoiler_fraction: float,
    ) -> list[Sample]:
        if not candidates:
            return []

        n_clusters = max(1, min(k, len(candidates)))
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=self.random_state)
        labels = kmeans.fit_predict(candidate_emb)

        centers = kmeans.cluster_centers_
        center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
        center_norms[center_norms == 0] = 1.0
        centers = centers / center_norms  # candidate_emb is already L2-normalized

        sim_matrix = candidate_emb @ prior_emb.T
        prior_relevance = sim_matrix.max(axis=1)
        nearest_prior_idx = sim_matrix.argmax(axis=1)

        safe_mask = position_fractions <= spoiler_fraction

        taste_samples: list[Sample] = []
        representative_samples: list[Sample] = []

        for cluster_id in range(n_clusters):
            in_cluster_safe = np.where((labels == cluster_id) & safe_mask)[0]
            if len(in_cluster_safe) == 0:
                continue  # this theme has no safe (early-book) example to show

            centroid_sims = candidate_emb[in_cluster_safe] @ centers[cluster_id]
            best_repr = in_cluster_safe[int(np.argmax(centroid_sims))]
            representative_samples.append(
                Sample(
                    chunk=candidates[best_repr],
                    score=float(centroid_sims.max()),
                    rank=0,
                    kind="representative",
                    position_fraction=float(position_fractions[best_repr]),
                )
            )

            best_taste = in_cluster_safe[int(np.argmax(prior_relevance[in_cluster_safe]))]
            taste_samples.append(
                Sample(
                    chunk=candidates[best_taste],
                    score=float(prior_relevance[best_taste]),
                    rank=0,
                    kind="taste",
                    matched_prior_text=priors[nearest_prior_idx[best_taste]].text,
                    position_fraction=float(position_fractions[best_taste]),
                )
            )

        for group in (taste_samples, representative_samples):
            group.sort(key=lambda s: s.score, reverse=True)
            for rank, s in enumerate(group, start=1):
                s.rank = rank

        return taste_samples + representative_samples
