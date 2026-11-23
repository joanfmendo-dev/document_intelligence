"""DocBase: the web prototype. Upload a document -> review the extracted fields next to it -> save them in the
database; browse, edit and delete saved records. The uploaded file is only kept while it is being reviewed.

    from docintel.web import create_app
    app = create_app()          # or: python -m docintel.web

Pages are rendered on the server (Jinja2); the little JavaScript in static/app.js only adds drag & drop,
the lab-rows buttons and Esc on the delete dialog. The routes live in auth, upload_routes, record_routes and copilot_routes;
what they share (database, models, templates, what to do after a write) is in site.Site.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .auth import COOKIE, OPEN_PATHS, auth_routes
from .copilot_routes import copilot_routes
from .record_routes import record_routes
from .site import HERE, Site
from .upload_routes import upload_routes

LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"]


def create_app(*, db_path=None, uploads_dir=None, pipeline: Callable | None = None, allowed_hosts: list[str] | None = None,
               max_upload_mb: int | None = None, catalog_path=None, copilot_llm=None, embedder=None, users: dict[str, str] | None = None) -> FastAPI:
    """Build the app. Every argument defaults to the matching setting (docintel.shared.settings); `pipeline` is
    (path, on_step) -> Extraction, and is swapped in tests so they need neither OCR nor a model.
    `copilot_llm` and `embedder` replace the language and embedding models (tests); `catalog_path` is the copilot's catalog file,
    which only this app writes (by default next to the database). `users` ({name: hash}) are the people who may sign in;
    by default they come from DOCINTEL_USERS and the app does not start without them. `users={}` switches the sign-in off (tests only)."""
    site = Site(db_path=db_path, uploads_dir=uploads_dir, pipeline=pipeline, max_upload_mb=max_upload_mb,
                catalog_path=catalog_path, copilot_llm=copilot_llm, embedder=embedder, users=users)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        site.uploads.prepare()
        site.open_db().close()  # creates the table; fails now, and clearly, if another process has the file open
        yield
        site.jobs.shutdown()

    app = FastAPI(title="DocBase", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def protect(request: Request, call_next):
        # A page on another site must not be able to POST here (delete, overwrite) while this one is open.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and urlsplit(origin).netloc != request.headers.get("host"):
            return PlainTextResponse("Origin not allowed.", status_code=403)
        request.state.user = site.auth.user_of(request.cookies.get(COOKIE))
        if site.auth.enabled and not request.state.user and not request.url.path.startswith(OPEN_PATHS):
            return site.auth.deny(request)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"  # these are medical records: keep them out of the browser cache
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return site.render(request, "error.html", exc.status_code, active="", title=f"Error {exc.status_code}", message=exc.detail)

    for routes in (auth_routes, upload_routes, record_routes, copilot_routes):
        app.include_router(routes(site))
    return app
