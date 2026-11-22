"""Upload a document, follow its extraction, review the fields next to it and save them."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.datastructures import FormData

from ..database import RecordExistsError
from . import forms
from .site import Site, found, go, read_form
from .storage import DOCUMENT_SUFFIXES, FileTooLarge, kind_of

STEP_ORDER = ["reading", "extracting", "coding", "done"]
STEP_LABELS = ["Reading the text (OCR on images)", "Extracting the fields", "Assigning the ICD-10 code", "Review and save"]


def steps_of(job) -> list[tuple[str, str]]:
    failed = job.step == "error"
    current = STEP_ORDER.index(job.failed_in if failed else job.step)
    state = lambda i: "done" if i < current else ("failed" if failed else "in progress") if i == current else "pending"
    return [(label, state(i)) for i, label in enumerate(STEP_LABELS)]


def file_response(path: Path) -> FileResponse:
    kind = kind_of(path)
    media = "text/plain; charset=utf-8" if kind == "text" else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name, headers={"X-Content-Type-Options": "nosniff"},
                        content_disposition_type="attachment" if kind == "other" else "inline")


def upload_routes(site: Site) -> APIRouter:
    router = APIRouter()
    errors_by_code = {
        "type": "Unsupported file type. Use a PDF, Word (.docx), an image or a text file.",
        "empty": "Choose a file (it cannot be empty).",
        "size": f"The file is larger than the {site.max_mb} MB limit.",
    }

    def upload_context(job=None, **extra) -> dict:
        defaults = dict(active="upload", job=job, steps=steps_of(job) if job else [], max_mb=site.max_mb, error="", viewer=None,
                        accept=",".join(sorted(DOCUMENT_SUFFIXES)))
        return {**defaults, **extra}

    def review_context(job, values=None, errors=None, **extra) -> dict:
        """The review form of a finished job. A value still equal to what the model extracted and that was not
        found in the text stays flagged; once the user changes it, it counts as reviewed."""
        original = forms.to_values(job.extraction.fields)
        values = values or original
        now, then = forms.flat(values), forms.flat(original)
        flagged = [p for p in job.extraction.ungrounded if now.get(p) == then.get(p)]
        review = dict(values=values, errors=errors or {}, flags=set(flagged), flagged_labels=[forms.label_of(p) for p in flagged],
                      viewer={"url": f"/upload/{job.id}/document", "kind": kind_of(job.path), "name": job.filename},
                      pages=job.extraction.pages, notice=job.extraction.notice, mode="upload", replace=False, conflict=False)
        return upload_context(job, **{**review, **extra})

    @router.get("/")
    def home():
        return go("/upload")

    @router.get("/upload")
    def upload_page(request: Request, error: str = ""):
        return site.render(request, "upload.html", **upload_context(error=errors_by_code.get(error, "")))

    @router.post("/upload")
    def upload(file: UploadFile | None = File(None)):
        if file is None or not file.filename:
            return go("/upload?error=empty")
        name = Path(file.filename.replace("\\", "/")).name
        if Path(name).suffix.lower() not in DOCUMENT_SUFFIXES:
            return go("/upload?error=type")
        job_id = site.jobs.new_id()
        try:
            path = site.uploads.save(job_id, file.filename, file.file, site.max_mb * 1024 * 1024)
        except FileTooLarge:
            return go("/upload?error=size")
        except ValueError:
            return go("/upload?error=empty")
        site.jobs.start(job_id, name, path)
        return go(f"/upload/{job_id}")

    @router.get("/upload/{job_id}")
    def job_page(request: Request, job_id: str):
        job = found(site.jobs.get(job_id), "upload")
        return site.render(request, "upload.html", **(review_context(job) if job.step == "done" else upload_context(job)))

    @router.get("/upload/{job_id}/document")
    def job_document(job_id: str):
        return file_response(found(site.jobs.get(job_id), "document").path)

    @router.post("/upload/{job_id}/retry")
    def job_retry(job_id: str):
        site.jobs.retry(job_id)
        return go(f"/upload/{job_id}")

    @router.post("/upload/{job_id}/cancel")
    def job_cancel(job_id: str):
        if job := site.jobs.remove(job_id):
            site.uploads.discard(job.path)
        return go("/upload")

    @router.post("/upload/{job_id}/save")
    def job_save(request: Request, job_id: str, form: FormData = Depends(read_form)):
        job = found(site.jobs.get(job_id), "upload")
        if job.step != "done":
            return go(f"/upload/{job_id}")
        values = forms.from_form(form)
        fields, errors = forms.parse(values)
        replace, conflict = form.get("replace") == "on", False
        if not errors:
            try:
                with site.open_db() as db:
                    if replace and db.read_record(fields["record_id"]) is not None:
                        db.update_record(fields, keep_existing=False)
                    else:
                        db.create_record(fields)
                    site.after_write(db, fields["record_id"])
            except RecordExistsError:
                errors["record_id"] = "A record with this ID already exists. Change the ID or tick “Replace the existing record”."
                conflict = True
        if errors:
            return site.render(request, "upload.html", 422, **review_context(job, values, errors, replace=replace, conflict=conflict))
        site.uploads.discard(job.path)  # the data is in the database: the file is not needed any more
        site.jobs.remove(job_id)
        return go(f"/records?{urlencode({'ok': 'saved', 'id': fields['record_id']})}")

    return router
