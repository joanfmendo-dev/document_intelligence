"""DocBase: the web prototype. Upload a document -> review the extracted fields next to it -> save them in the
database; browse, edit and delete saved records. The uploaded file is only kept while it is being reviewed.

    from docintel.web import create_app
    app = create_app()          # or: python -m docintel.web

Pages are rendered on the server (Jinja2); the little JavaScript in static/app.js only adds drag & drop,
the lab-rows buttons and Esc on the delete dialog.
"""
from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from math import ceil
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlsplit

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.templating import Jinja2Templates

from ..db import RecordDB, RecordExistsError, RecordNotFoundError
from ..settings import s
from . import forms, pipeline as default_pipeline
from .jobs import Jobs
from .storage import DOCUMENT_SUFFIXES, UploadStore, FileTooLarge, kind_of

HERE = Path(__file__).parent
PAGE_SIZE = 25  # the default; the user can pick another from PAGE_SIZES
PAGE_SIZES = (20, 25, 50, 100)
LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
STEP_LABELS = ["Reading the text (OCR on images)", "Extracting the fields", "Review and save"]
FLASH = {"saved": "Record {id} saved.", "updated": "Changes to {id} saved.", "deleted": "Record {id} deleted."}


def create_app(*, db_path: str | Path | None = None, uploads_dir: str | Path | None = None,
               pipeline: Callable | None = None, allowed_hosts: list[str] | None = None,
               max_upload_mb: int | None = None) -> FastAPI:
    """Build the app. Every argument defaults to the matching setting (docintel.settings); `pipeline` is
    (path, on_step) -> Extraction, and is swapped in tests so they need neither OCR nor a model."""
    db_path = str(db_path or s.DB_PATH)
    max_mb = max_upload_mb or s.MAX_UPLOAD_MB
    uploads = UploadStore(uploads_dir or s.UPLOADS_DIR)
    jobs = Jobs(pipeline or default_pipeline.run)
    upload_errors = {
        "type": "Unsupported file type. Use a PDF, Word (.docx), an image or a text file.",
        "empty": "Choose a file (it cannot be empty).",
        "size": f"The file is larger than the {max_mb} MB limit.",
    }

    def open_db() -> RecordDB:
        return RecordDB(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        uploads.prepare()
        open_db().close()  # creates the table; fails now, and clearly, if another process has the file open
        yield
        jobs.shutdown()

    app = FastAPI(title="DocBase", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(PATIENT=forms.PATIENT, HISTORY=forms.HISTORY, EXAM=forms.EXAM, FINAL=forms.FINAL, VITALS=forms.VITALS,
                                 LAB_COLUMNS=forms.LAB_COLUMNS, LAB_TESTS=forms.LAB_TESTS, LAB_FLAGS=forms.LAB_FLAGS, SEX_CHOICES=forms.SEX_CHOICES, label_of=forms.label_of)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        # A page on another site must not be able to POST here (delete, overwrite) while this one is open.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and urlsplit(origin).netloc != request.headers.get("host"):
            return PlainTextResponse("Origin not allowed.", status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"  # these are medical records: keep them out of the browser cache
        return response

    # ------------------------------------------------------------ helpers
    def render(request: Request, name: str, status: int = 200, **context):
        context.setdefault("flash", FLASH.get(request.query_params.get("ok", ""), "").format(id=request.query_params.get("id", "")))
        return templates.TemplateResponse(request, name, context, status_code=status)

    def go(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    async def read_form(request: Request) -> FormData:
        return await request.form()

    def file_response(path: Path) -> FileResponse:
        kind = kind_of(path)
        media = "text/plain; charset=utf-8" if kind == "text" else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media, filename=path.name, headers={"X-Content-Type-Options": "nosniff"},
                            content_disposition_type="attachment" if kind == "other" else "inline")

    def found(value, what: str):
        """`value`, or a 404 page saying that `what` does not exist."""
        if value is None:
            raise HTTPException(404, f"The {what} you are looking for does not exist.")
        return value

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return render(request, "error.html", exc.status_code, active="", title=f"Error {exc.status_code}", message=exc.detail)

    # ------------------------------------------------------------ upload a document
    def steps_of(job) -> list[tuple[str, str]]:
        order = ["reading", "extracting", "done"]
        failed = job.step == "error"
        current = order.index(job.failed_in if failed else job.step)
        state = lambda i: "done" if i < current else ("failed" if failed else "in progress") if i == current else "pending"
        return [(label, state(i)) for i, label in enumerate(STEP_LABELS)]

    def upload_context(job=None, **extra) -> dict:
        defaults = dict(active="upload", job=job, steps=steps_of(job) if job else [], max_mb=max_mb, error="", viewer=None,
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
                      pages=job.extraction.pages, mode="upload", replace=False, conflict=False)
        return upload_context(job, **{**review, **extra})

    @app.get("/")
    def home():
        return go("/upload")

    @app.get("/upload")
    def upload_page(request: Request, error: str = ""):
        return render(request, "upload.html", **upload_context(error=upload_errors.get(error, "")))

    @app.post("/upload")
    def upload(file: UploadFile | None = File(None)):
        if file is None or not file.filename:
            return go("/upload?error=empty")
        if Path(file.filename.replace("\\", "/")).suffix.lower() not in DOCUMENT_SUFFIXES:
            return go("/upload?error=type")
        job_id = jobs.new_id()
        try:
            path = uploads.save(job_id, file.filename, file.file, max_mb * 1024 * 1024)
        except FileTooLarge:
            return go("/upload?error=size")
        except ValueError:
            return go("/upload?error=empty")
        jobs.start(job_id, Path(file.filename.replace("\\", "/")).name, path)
        return go(f"/upload/{job_id}")

    @app.get("/upload/{job_id}")
    def job_page(request: Request, job_id: str):
        job = found(jobs.get(job_id), "upload")
        context = review_context(job) if job.step == "done" else upload_context(job)
        return render(request, "upload.html", **context)

    @app.get("/upload/{job_id}/document")
    def job_document(job_id: str):
        return file_response(found(jobs.get(job_id), "document").path)

    @app.post("/upload/{job_id}/retry")
    def job_retry(job_id: str):
        jobs.retry(job_id)
        return go(f"/upload/{job_id}")

    @app.post("/upload/{job_id}/cancel")
    def job_cancel(job_id: str):
        if job := jobs.remove(job_id):
            uploads.discard(job.path)
        return go("/upload")

    @app.post("/upload/{job_id}/save")
    def job_save(request: Request, job_id: str, form: FormData = Depends(read_form)):
        job = found(jobs.get(job_id), "upload")
        if job.step != "done":
            return go(f"/upload/{job_id}")
        values = forms.from_form(form)
        fields, errors = forms.parse(values)
        replace, conflict = form.get("replace") == "on", False
        if not errors:
            try:
                with open_db() as db:
                    if replace and db.read_record(fields["record_id"]) is not None:
                        db.update_record(fields, keep_existing=False)
                    else:
                        db.create_record(fields)
            except RecordExistsError:
                errors["record_id"] = "A record with this ID already exists. Change the ID or tick “Replace the existing record”."
                conflict = True
        if errors:
            return render(request, "upload.html", 422, **review_context(job, values, errors, replace=replace, conflict=conflict))
        uploads.discard(job.path)  # the data is in the database: the file is not needed any more
        jobs.remove(job_id)
        return go(f"/records?{urlencode({'ok': 'saved', 'id': fields['record_id']})}")

    # ------------------------------------------------------------ the database
    def list_context(q: str, department: str, page: int, per_page: int = PAGE_SIZE) -> dict:
        per_page = per_page if per_page in PAGE_SIZES else PAGE_SIZE
        page = max(page, 1)
        with open_db() as db:
            departments = db.departments()
            rows, total = db.list_records(search=q, department=department, limit=per_page, offset=(page - 1) * per_page)
            pages = max(ceil(total / per_page), 1)
            if page > pages:  # a stale link to a page that no longer exists
                page = pages
                rows, total = db.list_records(search=q, department=department, limit=per_page, offset=(page - 1) * per_page)

        def query(n: int) -> str:  # the filters, the page size and page n; defaults are left out to keep links short
            params = {"q": q, "d": department, "n": per_page if per_page != PAGE_SIZE else "", "page": n if n > 1 else ""}
            return urlencode({k: v for k, v in params.items() if v})

        link = lambda n: "/records" + ("?" + query(n) if query(n) else "")
        return dict(active="records", rows=rows, total=total, q=q, d=department, departments=departments, page=page, pages=pages,
                    per_page=per_page, page_sizes=PAGE_SIZES, start=(page - 1) * per_page + 1 if rows else 0, end=(page - 1) * per_page + len(rows),
                    first_url=link(1) if page > 1 else "", prev_url=link(page - 1) if page > 1 else "",
                    next_url=link(page + 1) if page < pages else "", last_url=link(pages) if page < pages else "",
                    here=link(page), keep=query(page), confirm=None)

    @app.get("/records")
    def records_page(request: Request, q: str = "", d: str = "", page: int = 1, n: int = PAGE_SIZE):
        return render(request, "records.html", **list_context(q.strip(), d, page, n))

    @app.get("/records/{record_id}/delete")
    def delete_page(request: Request, record_id: str, q: str = "", d: str = "", page: int = 1, n: int = PAGE_SIZE):
        with open_db() as db:
            record = db.read_record(record_id)
        found(record, "record")
        context = list_context(q.strip(), d, page, n)
        context["confirm"] = {"id": record_id, "patient": record["patient_name"]}
        return render(request, "records.html", **context)

    @app.post("/records/{record_id}/delete")
    def delete(record_id: str, form: FormData = Depends(read_form)):
        with open_db() as db:
            deleted = db.delete_record(record_id)
        keep = {k: form.get(k) for k in ("q", "d", "n", "page") if form.get(k)}
        return go("/records?" + urlencode({**keep, "ok": "deleted" if deleted else "", "id": record_id}))

    # ------------------------------------------------------------ edit a record
    def edit_context(record_id: str, values: dict, errors: dict) -> dict:
        return dict(active="records", mode="edit", record_id=record_id, values=values, errors=errors, flags=set(), viewer=None, replace=False, conflict=False)

    @app.get("/records/{record_id}/edit")
    def edit_page(request: Request, record_id: str):
        with open_db() as db:
            record = db.read_record(record_id)
        found(record, "record")
        return render(request, "edit.html", **edit_context(record_id, forms.to_values(record), {}))

    @app.post("/records/{record_id}/edit")
    def edit_save(request: Request, record_id: str, form: FormData = Depends(read_form)):
        values = forms.from_form(form)
        values["record_id"] = record_id  # the ID is the key: it cannot be changed here
        fields, errors = forms.parse(values)
        if errors:
            return render(request, "edit.html", 422, **edit_context(record_id, values, errors))
        try:
            with open_db() as db:
                db.update_record(fields, keep_existing=False)  # a field the user emptied is cleared
        except RecordNotFoundError:
            found(None, "record")
        return go(f"/records?{urlencode({'ok': 'updated', 'id': record_id})}")

    return app
