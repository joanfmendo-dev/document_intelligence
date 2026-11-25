"""Running on a server: the Blob Storage copy of the database, the health check, the page limit and listening beyond this computer."""
import pytest
from azure.core.exceptions import ResourceNotFoundError
from fastapi.testclient import TestClient

from docintel.shared.settings import s as settings
from docintel.web import create_app
from docintel.web.__main__ import main
from docintel.web.blob import BlobCopy
from docintel.web.pipeline import run


class FakeContainer:
    def __init__(self, blobs=None):
        self.blobs = dict(blobs or {})

    def download_blob(self, name):
        if name not in self.blobs:
            raise ResourceNotFoundError("missing")
        return type("Blob", (), {"readall": lambda _: self.blobs[name]})()

    def upload_blob(self, name, data, overwrite=False):
        assert overwrite
        self.blobs[name] = data


def test_pull_downloads_what_exists_and_push_uploads_what_is_on_disk(tmp_path):
    db, catalog = tmp_path / "a" / "db.duckdb", tmp_path / "a" / "db.catalog.yaml"
    cloud = FakeContainer({"db.duckdb": b"master"})
    copy = BlobCopy(cloud, [db, catalog])
    copy.pull()
    assert db.read_bytes() == b"master" and not catalog.exists()  # the catalog was not in the container yet
    catalog.write_text("x: 1")
    copy.push()
    assert cloud.blobs == {"db.duckdb": b"master", "db.catalog.yaml": b"x: 1"}


def test_the_app_downloads_the_database_at_start(tmp_path):
    from docintel.database import RecordDB
    seed = tmp_path / "seed.duckdb"
    RecordDB(seed).close()
    work = tmp_path / "work" / "db.duckdb"
    app = create_app(db_path=work, uploads_dir=tmp_path / "up", allowed_hosts=["testserver"], users={},
                     blob=BlobCopy(FakeContainer({"db.duckdb": seed.read_bytes()}), [work]))
    with TestClient(app) as client:
        assert work.is_file() and client.get("/records").status_code == 200


def test_a_save_uploads_the_database_and_a_failed_upload_does_not_fail_the_save(tmp_path, caplog):
    from docintel.web.site import Site
    work = tmp_path / "db.duckdb"
    cloud = FakeContainer()
    site = Site(db_path=work, uploads_dir=tmp_path / "up", users={}, blob=BlobCopy(cloud, [work]), embedder=object())
    with site.open_db() as db:
        site.after_write(db, None)
        assert cloud.blobs["db.duckdb"].startswith(b"") and len(cloud.blobs["db.duckdb"]) > 0
        cloud.upload_blob = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        site.after_write(db, None)  # does not raise
    assert "could not upload the database" in caplog.text


def test_health_check_needs_no_sign_in(tmp_path):
    from docintel.web import auth
    app = create_app(db_path=tmp_path / "db.duckdb", uploads_dir=tmp_path / "up", allowed_hosts=["testserver"], users={"maria": auth.hash_password("x")})
    with TestClient(app) as client:
        assert client.get("/healthz").text == "ok" and client.get("/records", follow_redirects=False).status_code == 303


def test_the_page_limit_refuses_long_pdfs_before_reading_them(tmp_path, monkeypatch):
    import pypdfium2
    pdf = pypdfium2.PdfDocument.new()
    for _ in range(3):
        pdf.new_page(200, 200)
    path = tmp_path / "long.pdf"
    pdf.save(path)
    monkeypatch.setattr(settings, "MAX_PAGES", 2)
    with pytest.raises(ValueError, match="3 pages; the limit is 2"):
        run(path, lambda step: None)


def test_listening_beyond_this_computer_needs_allowed_hosts(monkeypatch):
    monkeypatch.setattr(settings, "USERS", {"maria": "a$b"})
    monkeypatch.setattr(settings, "ALLOWED_HOSTS", [])
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])
