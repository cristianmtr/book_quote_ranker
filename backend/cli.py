"""Dev-verification entrypoint. Calls the exact same pipeline.process_books()
used by the FastAPI app, so this doubles as a manual sanity-check tool.

Usage:
    python -m backend.cli --candidates a.epub b.epub --priors data/highlights.md --k 3
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from .pipeline import process_books


def main() -> None:
    parser = argparse.ArgumentParser(description="BookQuotes pipeline dev-verification CLI")
    parser.add_argument("--candidates", nargs="+", required=True, help="EPUB file paths")
    parser.add_argument("--priors", required=True, help="Priors markdown file path")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out-dir", default="data/output/manual")
    args = parser.parse_args()

    candidate_paths = [(str(uuid.uuid4()), Path(p), Path(p).name) for p in args.candidates]

    def on_status(book_id: str, status: str) -> None:
        print(f"[{book_id[:8]}] {status}")

    results = process_books(candidate_paths, Path(args.priors), args.k, Path(args.out_dir), on_status)

    for r in results:
        print(f"\n=== {r.filename} ({r.status}) ===")
        if r.error:
            print("ERROR:", r.error)
            continue
        for s in r.samples:
            preview = s.chunk.text[:150].replace("\n", " ")
            print(f"  rank={s.rank} score={s.score:.3f} words={len(s.chunk.text.split())} :: {preview}...")
        print(f"  -> {r.output_epub_path}")


if __name__ == "__main__":
    main()
