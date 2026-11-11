"""python -m docintel.web  ->  DocBase on http://127.0.0.1:8000"""
from __future__ import annotations

import argparse

from ..settings import s

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m docintel.web", description="DocBase: load documents, review their fields, keep them in a local database.")
    parser.add_argument("--host", default="127.0.0.1", help="only this computer can reach the app (it has no login): 127.0.0.1 or localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK:
        parser.error("the app has no login yet, so it only listens on this computer (127.0.0.1 / localhost)")

    import duckdb
    import uvicorn

    from ..db import RecordDB
    from .app import create_app

    try:
        RecordDB(s.DB_PATH).close()
    except duckdb.IOException as exc:
        raise SystemExit(f"Cannot open the database {s.DB_PATH!r}: another program (a notebook, another DocBase) has it open.\n{exc}")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
