# BookQuotes

Upload a set of EPUB novels ("Candidates") and a Markdown file of quotes you
liked from other books ("Priors"). BookQuotes extracts the top-K passages per
Candidate that read most like your Priors ("Samples"), and lets you download
one merged EPUB of Samples per Candidate.

## Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
```

The Torch line is separate on purpose: a plain `pip install` resolves Torch to
its default CUDA build on Windows (a multi-GB download). The CPU-only wheel
above is all this app needs — embedding a few thousand short text chunks with
MiniLM is fast on CPU.

## Run the app

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000/, upload your Candidates + Priors, set K, click
Process. The first run downloads the embedding model (`all-MiniLM-L6-v2`,
~90MB) from Hugging Face and caches it locally.

## CLI (no browser needed)

Runs the exact same pipeline as the web app — useful for quick sanity checks:

```powershell
.venv\Scripts\python -m backend.cli --candidates book1.epub book2.epub --priors data/highlights.md --k 3
```

## Smoke tests

```powershell
.venv\Scripts\python scripts\smoke_test.py
```

## Project layout

```
backend/
  main.py         FastAPI app + routes
  pipeline.py     end-to-end orchestration (parse -> chunk -> embed -> select -> assemble)
  chunking.py     paragraph/sentence-aware chunking (swap via Chunker Protocol)
  embeddings.py   embedding model wrapper (swap via EmbeddingModel Protocol)
  selection.py    nearest-Prior + MMR diversity selection (swap via Selector Protocol)
  epub_io.py      EPUB <-> Markdown conversion
  priors.py       Priors markdown parsing
  cli.py          dev-verification entrypoint
frontend/         plain HTML/CSS/JS, no build step
data/             uploads, job state, and output EPUBs (gitignored, except highlights.md)
```

## Swapping models/algorithms

Each pluggable piece is a small `typing.Protocol` with one default
implementation, dispatched from a plain dict in `config.py`/module scope —
no plugin framework. To try a different embedding model, add an entry to
`_EMBEDDERS` in `embeddings.py`; same pattern for `chunking.py` /
`selection.py`.
