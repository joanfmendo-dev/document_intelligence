"""Reading a document and extracting its fields takes a while (OCR, then a model), so it runs in a background
thread while the page polls. Jobs live in memory: restarting the app forgets the ones not yet saved."""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .pipeline import Extraction

log = logging.getLogger("docintel.web")


@dataclass
class Job:
    id: str
    filename: str  # as the user named it
    path: Path  # where the upload is
    step: str = "reading"  # reading -> extracting -> coding -> done | error
    failed_in: str = ""  # the step that was running when it failed
    extraction: Extraction | None = None
    error: str = ""

    @property
    def running(self) -> bool:
        return self.step in ("reading", "extracting", "coding")


class Jobs:
    def __init__(self, pipeline: Callable[[Path, Callable[[str], None]], Extraction]):
        self._pipeline = pipeline
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docintel-job")  # one model call at a time

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def start(self, job_id: str, filename: str, path: Path) -> Job:
        job = Job(job_id, filename, path)
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def retry(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job and job.step == "error":
            job.step, job.failed_in, job.error = "reading", "", ""
            self._pool.submit(self._run, job)
        return job

    def remove(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job) -> None:
        def on_step(step: str) -> None:
            job.step = step

        try:
            job.extraction = self._pipeline(job.path, on_step)
            job.step = "done"
        except Exception as exc:  # whatever went wrong must reach the page, not die in the thread
            log.exception("Processing %s failed", job.filename)
            job.failed_in, job.error = job.step, f"{type(exc).__name__}: {exc}"[:600]
            job.step = "error"
