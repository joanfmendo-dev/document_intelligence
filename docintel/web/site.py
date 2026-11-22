"""What every page of the app shares: where the data lives, the models, the templates, and what has to happen after a write."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from fastapi.responses import RedirectResponse
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.templating import Jinja2Templates

from ..copilot.catalog import Catalog
from ..database import RecordDB
from ..shared.embeddings import EmbeddingError
from ..shared.settings import s
from . import forms, pipeline as default_pipeline
from .chat import ChatStore
from .jobs import Jobs
from .storage import UploadStore

HERE = Path(__file__).parent
FLASH = {"saved": "Record {id} saved.", "updated": "Changes to {id} saved.", "deleted": "Record {id} deleted."}
log = logging.getLogger(__name__)


async def read_form(request: Request) -> FormData:
    return await request.form()


def go(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def found(value, what: str):
    """`value`, or a 404 page saying that `what` does not exist."""
    if value is None:
        raise HTTPException(404, f"The {what} you are looking for does not exist.")
    return value


class Site:
    def __init__(self, *, db_path=None, uploads_dir=None, pipeline: Callable | None = None, max_upload_mb: int | None = None,
                 catalog_path=None, copilot_llm=None, embedder=None):
        self.db_path = str(db_path or s.DB_PATH)
        self.catalog_file = Path(catalog_path) if catalog_path else Path(self.db_path).with_suffix(".catalog.yaml")
        self.max_mb = max_upload_mb or s.MAX_UPLOAD_MB
        self.uploads = UploadStore(uploads_dir or s.UPLOADS_DIR)
        self.jobs = Jobs(pipeline or default_pipeline.run)
        self.copilot_llm, self.embedder = copilot_llm, embedder
        self.chats = ChatStore()
        self.templates = Jinja2Templates(directory=HERE / "templates")
        self.templates.env.globals.update(
            {name: getattr(forms, name) for name in ("PATIENT", "ICD", "HISTORY", "EXAM", "FINAL", "VITALS", "LAB_COLUMNS", "LAB_TESTS",
                                                      "LAB_FLAGS", "SEX_CHOICES", "label_of")})

    def open_db(self) -> RecordDB:
        return RecordDB(self.db_path, embedder=self.embedder)

    def render(self, request: Request, name: str, status: int = 200, **context):
        context.setdefault("flash", FLASH.get(request.query_params.get("ok", ""), "").format(id=request.query_params.get("id", "")))
        return self.templates.TemplateResponse(request, name, context, status_code=status)

    def after_write(self, db: RecordDB, record_id: str | None) -> None:
        """Keep what the copilot reads up to date: the record's vector and the catalog of values. Neither may fail a save:
        a record without a vector is picked up by `embed_pending`, and a stale catalog falls back to the live database."""
        if record_id:
            try:
                db.embed_record(record_id)
            except EmbeddingError as exc:
                log.warning("record %s saved without embedding: %s", record_id, exc)
        try:
            file = self.catalog_file
            previous = Catalog.from_yaml(file) if file.is_file() else None
            (previous.refresh(db.con) if previous else Catalog.build(db.con)).to_yaml(file)
        except Exception as exc:  # never lose a save over the catalog
            log.warning("could not update the catalog: %s", exc)
