"""python -m docintel.web  ->  Medical Records on http://127.0.0.1:8000"""
from __future__ import annotations

import argparse

from ..shared.settings import s
from .auth import NO_USERS

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m docintel.web", description="Medical Records: load documents, review their fields, keep them in a local database.")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 or localhost (only this computer); another address (0.0.0.0) needs ALLOWED_HOSTS")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK and not s.ALLOWED_HOSTS:
        parser.error("to listen beyond this computer, set ALLOWED_HOSTS (the names the app answers to, or *)")

    if not s.USERS:
        raise SystemExit(NO_USERS)

    import duckdb
    import uvicorn

    from ..database import RecordDB
    from .app import create_app

    if not s.BLOB_CONNECTION_STRING:  # with a Blob copy the app downloads the database itself at start, and nobody else has the file
        try:
            RecordDB(s.DB_PATH).close()
        except duckdb.IOException as exc:
            raise SystemExit(f"Cannot open the database {s.DB_PATH!r}: another program (a notebook, another Medical Records) has it open.\n{exc}") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="*" if args.host not in LOOPBACK else "127.0.0.1")


if __name__ == "__main__":
    main()
