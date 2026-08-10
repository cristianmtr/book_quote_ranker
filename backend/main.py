from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_K, SPOILER_GUARD_FRACTION
from .embeddings import get_embedder
from .job_store import job_store
from .models import JobStatus
from .pipeline import process_books

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/output")

_state: dict = {}
_process_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["embedder"] = get_embedder()  # load model once, reuse across jobs
    yield


app = FastAPI(title="BookQuotes", lifespan=lifespan)


@app.post("/api/jobs", response_model=JobStatus)
async def create_job(
    candidates: list[UploadFile] = File(...),
    priors: UploadFile = File(...),
    k: int = Form(DEFAULT_K),
    spoiler_fraction: float = Form(SPOILER_GUARD_FRACTION),
) -> JobStatus:
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    candidates_dir = job_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    priors_path = job_dir / "priors.md"
    priors_path.write_bytes(await priors.read())

    book_entries: list[tuple[str, str]] = []
    candidate_paths: list[tuple[str, Path, str]] = []
    for f in candidates:
        book_id = str(uuid.uuid4())
        safe_name = Path(f.filename or "book.epub").name
        dest = candidates_dir / f"{book_id}_{safe_name}"
        dest.write_bytes(await f.read())
        book_entries.append((book_id, safe_name))
        candidate_paths.append((book_id, dest, safe_name))

    job_store.register_candidates(job_id, candidate_paths)
    return job_store.create(job_id, k, spoiler_fraction, book_entries)


def _run_job(job_id: str) -> None:
    job = job_store.get(job_id)
    candidates = job_store.get_candidates(job_id)
    priors_path = UPLOAD_DIR / job_id / "priors.md"
    out_dir = OUTPUT_DIR / job_id

    def on_status(book_id: str, status: str) -> None:
        job_store.set_book_status(job_id, book_id, status)

    try:
        with _process_lock:  # serialize jobs against the shared embedding model
            results = process_books(
                candidates, priors_path, job.k, out_dir, on_status,
                embedder=_state["embedder"], spoiler_fraction=job.spoiler_fraction,
            )
        for result in results:
            job_store.set_book_result(job_id, result)
        final_state = "error" if all(r.error for r in results) else "done"
        job_store.set_state(job_id, final_state)
    except Exception as exc:  # noqa: BLE001 - surface as job-level error, e.g. empty Priors file
        job_store.set_state(job_id, "error", error=str(exc))


@app.post("/api/jobs/{job_id}/process", status_code=202)
def process_job(job_id: str, background_tasks: BackgroundTasks) -> dict:
    if job_store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    job_store.set_state(job_id, "processing")
    background_tasks.add_task(_run_job, job_id)
    return {"job_id": job_id, "state": "processing"}


@app.get("/api/jobs/{job_id}/status", response_model=JobStatus)
def get_status(job_id: str) -> JobStatus:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}/books/{book_id}/preview")
def get_preview(job_id: str, book_id: str) -> dict:
    result = job_store.get_book_result(job_id, book_id)
    if result is None:
        raise HTTPException(404, "book result not available yet")
    return {
        "book_id": book_id,
        "filename": result.filename,
        "status": result.status,
        "error": result.error,
        "aggregate_score": result.aggregate_score,
        "samples": [
            {
                "rank": s.rank,
                "score": s.score,
                "text": s.chunk.text,
                "matched_prior_text": s.matched_prior_text,
                "position_fraction": s.position_fraction,
                "window_label": s.chunk.window_label,
            }
            for s in result.samples
        ],
    }


@app.get("/api/jobs/{job_id}/books/{book_id}/download")
def download(job_id: str, book_id: str) -> FileResponse:
    result = job_store.get_book_result(job_id, book_id)
    if result is None or result.output_path is None:
        raise HTTPException(404, "output not ready")
    return FileResponse(
        result.output_path,
        filename=result.output_path.name,
        media_type="text/markdown",
    )


# Mounted last so the /api routes above take precedence over static serving.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
