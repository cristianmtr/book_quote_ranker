DEFAULT_K = 3
DEFAULT_EMBEDDER = "minilm"

# Samples are only ever returned from the first N% of each Candidate's text
# (positionally), so they never reveal later plot points (spoilers). Note
# clustering/scoring still scans the WHOLE book — this only restricts which
# chunks are eligible to be picked as output, not what's used to define the
# book's representative themes.
SPOILER_GUARD_FRACTION = 0.2
