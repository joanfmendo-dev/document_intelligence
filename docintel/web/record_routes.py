"""Browse, edit and delete the saved records."""
from __future__ import annotations

from math import ceil
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import FormData

from ..database import RecordNotFoundError
from ..database.data_transformation import ICD10
from . import forms
from .site import Site, found, go, read_form

PAGE_SIZE = 25  # the default; the user can pick another from PAGE_SIZES
PAGE_SIZES = (20, 25, 50, 100)


def record_routes(site: Site) -> APIRouter:
    router = APIRouter()

    def list_context(q: str, department: str, page: int, per_page: int = PAGE_SIZE) -> dict:
        per_page = per_page if per_page in PAGE_SIZES else PAGE_SIZE
        page = max(page, 1)
        with site.open_db() as db:
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

    def read_or_404(record_id: str) -> dict:
        with site.open_db() as db:
            return found(db.read_record(record_id), "record")

    def edit_context(record_id: str, values: dict, errors: dict) -> dict:
        return dict(active="records", mode="edit", record_id=record_id, values=values, errors=errors, flags=set(), viewer=None, replace=False, conflict=False)

    @router.get("/records")
    def records_page(request: Request, q: str = "", d: str = "", page: int = 1, n: int = PAGE_SIZE):
        return site.render(request, "records.html", **list_context(q.strip(), d, page, n))

    @router.get("/records/{record_id}/delete")
    def delete_page(request: Request, record_id: str, q: str = "", d: str = "", page: int = 1, n: int = PAGE_SIZE):
        record = read_or_404(record_id)
        context = list_context(q.strip(), d, page, n)
        context["confirm"] = {"id": record_id, "patient": record["patient_name"]}
        return site.render(request, "records.html", **context)

    @router.post("/records/{record_id}/delete")
    def delete(record_id: str, form: FormData = Depends(read_form)):
        with site.open_db() as db:
            deleted = db.delete_record(record_id)
            if deleted:
                site.after_write(db, None)
        keep = {k: form.get(k) for k in ("q", "d", "n", "page") if form.get(k)}
        return go("/records?" + urlencode({**keep, "ok": "deleted" if deleted else "", "id": record_id}))

    @router.get("/icd10")
    def icd10_lookup(code: str = ""):
        """The chapter and category of a code, for the form to show while the user types it ({} if it is not a code)."""
        coding = ICD10.resolve(code)
        return JSONResponse(coding._asdict() if coding else {})

    @router.get("/records/{record_id}/edit")
    def edit_page(request: Request, record_id: str):
        return site.render(request, "edit.html", **edit_context(record_id, forms.to_values(read_or_404(record_id)), {}))

    @router.post("/records/{record_id}/edit")
    def edit_save(request: Request, record_id: str, form: FormData = Depends(read_form)):
        values = forms.from_form(form)
        values["record_id"] = record_id  # the ID is the key: it cannot be changed here
        fields, errors = forms.parse(values)
        if errors:
            return site.render(request, "edit.html", 422, **edit_context(record_id, values, errors))
        try:
            with site.open_db() as db:
                db.update_record(fields, keep_existing=False)  # a field the user emptied is cleared
                site.after_write(db, record_id)
        except RecordNotFoundError:
            found(None, "record")
        return go(f"/records?{urlencode({'ok': 'updated', 'id': record_id})}")

    return router
