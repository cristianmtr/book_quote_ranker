from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import BookResult, BookResultSummary, JobStatus

JOBS_DIR = Path("data/jobs")


class JobStore:
    """In-memory job state + JSON write-through per job (survives restarts,
    trivially inspectable). No DB server needed at single-user local scale.
    Candidate file paths are kept in-memory only (job-lifetime); a restart
    loses in-flight processing state, which is acceptable for this tool."""

    def __init__(self, jobs_dir: Path = JOBS_DIR):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, JobStatus] = {}
        self._book_results: dict[str, dict[str, BookResult]] = {}
        self._candidates: dict[str, list[tuple[str, Path, str]]] = {}

    def create(
        self, job_id: str, k: int, spoiler_fraction: float, books: list[tuple[str, str]]
    ) -> JobStatus:
        now = datetime.now(timezone.utc)
        summaries = [
            BookResultSummary(book_id=book_id, filename=filename, status="pending", sample_count=0)
            for book_id, filename in books
        ]
        status = JobStatus(
            job_id=job_id,
            state="uploaded",
            k=k,
            spoiler_fraction=spoiler_fraction,
            books=summaries,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = status
            self._book_results[job_id] = {}
        self._write(status)
        return status

    def register_candidates(self, job_id: str, candidates: list[tuple[str, Path, str]]) -> None:
        with self._lock:
            self._candidates[job_id] = candidates

    def get_candidates(self, job_id: str) -> list[tuple[str, Path, str]]:
        with self._lock:
            return self._candidates.get(job_id, [])

    def set_state(self, job_id: str, state: str, error: str | None = None) -> None:
        with self._lock:
            status = self._jobs[job_id]
            status.state = state
            status.error = error
            status.updated_at = datetime.now(timezone.utc)
        self._write(status)

    def set_book_status(self, job_id: str, book_id: str, book_status: str) -> None:
        with self._lock:
            status = self._jobs[job_id]
            for b in status.books:
                if b.book_id == book_id:
                    b.status = book_status
                    break
            status.updated_at = datetime.now(timezone.utc)
        self._write(status)

    def set_book_result(self, job_id: str, result: BookResult) -> None:
        with self._lock:
            self._book_results[job_id][result.book_id] = result
            status = self._jobs[job_id]
            for b in status.books:
                if b.book_id == result.book_id:
                    b.status = result.status
                    b.sample_count = len(result.samples)
                    b.aggregate_score = result.aggregate_score
                    b.error = result.error
                    break
            status.updated_at = datetime.now(timezone.utc)
        self._write(status)

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_book_result(self, job_id: str, book_id: str) -> BookResult | None:
        with self._lock:
            return self._book_results.get(job_id, {}).get(book_id)

    def _write(self, status: JobStatus) -> None:
        path = self.jobs_dir / f"{status.job_id}.json"
        path.write_text(status.model_dump_json(indent=2), encoding="utf-8")


job_store = JobStore()
