DEFAULT_K = 3
DEFAULT_EMBEDDER = "minilm"
MMR_LAMBDA = 0.7

# Only chunk/select from the first N% of each Candidate's text, so Samples
# never come from later plot points (spoilers).
SPOILER_GUARD_FRACTION = 0.2


def pool_size(k: int) -> int:
    return max(30, 10 * k)
