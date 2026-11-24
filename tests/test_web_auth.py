"""The login page: who gets in, what the cookie is worth, and that nothing changes while no users are configured."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from docintel.shared.settings import s as settings, users_from
from docintel.web import auth, create_app, passwd
from docintel.web.__main__ import main
from docintel.web.pipeline import Extraction
from test_web import FIELDS

USERS = {"maria": auth.hash_password("correct horse"), "luis": auth.hash_password("battery staple")}


@pytest.fixture
def app(tmp_path):
    return create_app(db_path=tmp_path / "db.duckdb", uploads_dir=tmp_path / "up", allowed_hosts=["testserver"],
                      pipeline=lambda path, on_step: Extraction(FIELDS, [], 1), users=USERS)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def sign_in(client, user="maria", password="correct horse", **extra):
    return client.post("/login", data={"username": user, "password": password, **extra}, follow_redirects=False)


def test_the_app_does_not_start_without_users(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "USERS", {})
    with pytest.raises(RuntimeError, match="config.json"):
        create_app(db_path=tmp_path / "db.duckdb", uploads_dir=tmp_path / "up")
    with pytest.raises(SystemExit) as stop:  # python -m docintel.web: a clear message, and no server
        main([])
    assert "USERS" in str(stop.value) and "docintel.web.passwd" in str(stop.value)


def test_users_come_from_the_settings_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "USERS", USERS)
    with TestClient(create_app(db_path=tmp_path / "db.duckdb", uploads_dir=tmp_path / "up", allowed_hosts=["testserver"])) as c:
        assert c.get("/records", follow_redirects=False).headers["location"].startswith("/login")


def test_users_empty_switches_the_sign_in_off_for_tests(tmp_path):
    app = create_app(db_path=tmp_path / "db.duckdb", uploads_dir=tmp_path / "up", allowed_hosts=["testserver"], users={})
    with TestClient(app) as c:
        assert c.get("/records").status_code == 200
        assert "Sign out" not in c.get("/records").text


def test_a_page_sends_a_visitor_to_the_login_and_back_after_signing_in(client):
    r = client.get("/records?department=Cardiology", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=%2Frecords%3Fdepartment%3DCardiology"
    page = client.get(r.headers["location"])
    assert page.status_code == 200 and "Sign in" in page.text and 'value="/records?department=Cardiology"' in page.text
    assert 'href="/upload"' not in page.text  # the login page has no menu
    done = sign_in(client, next="/records?department=Cardiology")
    assert done.status_code == 303 and done.headers["location"] == "/records?department=Cardiology"
    assert "HttpOnly" in done.headers["set-cookie"] and "samesite=lax" in done.headers["set-cookie"].lower()
    page = client.get("/records")
    assert page.status_code == 200 and "maria" in page.text and "Sign out" in page.text


def test_static_files_and_the_login_are_open_but_requests_that_change_things_are_refused(client):
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/login").status_code == 200
    for method, url in (("post", "/copilot/ask"), ("post", "/copilot/clear"), ("post", "/upload"), ("post", "/records/X/delete")):
        r = getattr(client, method)(url, follow_redirects=False)
        assert r.status_code == 401, url


@pytest.mark.parametrize("user, password", [("maria", "wrong"), ("nobody", "correct horse"), ("maria", ""), ("luis", "correct horse")])
def test_wrong_credentials_are_refused_with_the_same_message(client, user, password):
    r = sign_in(client, user, password)
    assert r.status_code == 401 and "Wrong username or password." in r.text and "set-cookie" not in r.headers
    assert client.get("/records", follow_redirects=False).status_code == 303


def test_five_wrong_passwords_lock_that_user_for_a_while(client):
    for _ in range(auth.MAX_FAILURES):
        assert "Wrong username" in sign_in(client, password="x").text
    locked = sign_in(client)  # even the right password
    assert locked.status_code == 401 and "Too many failed attempts" in locked.text
    assert sign_in(client, "luis", "battery staple").status_code == 303  # another user is not affected


def test_the_lock_ends(monkeypatch):
    a = auth.Auth(USERS, "key")
    for _ in range(auth.MAX_FAILURES):
        assert a.sign_in("maria", "x")[0] is None
    assert "Too many" in a.sign_in("maria", "correct horse")[1]
    later = time.time() + auth.LOCK_SECONDS + 1
    monkeypatch.setattr(auth.time, "time", lambda: later)
    assert a.sign_in("maria", "correct horse")[0] is not None


def test_a_cookie_is_only_good_if_we_signed_it_and_it_has_not_expired(monkeypatch):
    a = auth.Auth(USERS, "key")
    token = a.token("maria")
    assert a.user_of(token) == "maria"
    assert auth.Auth(USERS, "other key").user_of(token) is None  # signed with another key
    assert a.user_of(token.replace("maria", "luis")) is None  # user changed, signature not
    assert a.user_of(token[:-1] + ("0" if token[-1] != "0" else "1")) is None
    assert a.user_of(None) is None and a.user_of("") is None and a.user_of("maria|9999999999|") is None and a.user_of("junk") is None
    assert auth.Auth({"luis": USERS["luis"]}, "key").user_of(token) is None  # the user was removed
    expired = time.time() + 8 * 3600 + 5
    monkeypatch.setattr(auth.time, "time", lambda: expired)
    assert a.user_of(token) is None  # expired


def test_a_forged_cookie_does_not_open_the_app(client):
    client.cookies.set(auth.COOKIE, "maria|9999999999|" + "0" * 64)
    assert client.get("/records", follow_redirects=False).status_code == 303


def test_signing_out_ends_the_session(client):
    sign_in(client)
    assert client.get("/records").status_code == 200
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/records", follow_redirects=False).status_code == 303


def test_a_signed_in_user_who_opens_the_login_goes_on_to_the_app(client):
    sign_in(client)
    r = client.get("/login?next=/copilot", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/copilot"


@pytest.mark.parametrize("target", ["https://evil.example/x", "//evil.example", "/\\evil.example", "javascript:alert(1)", "", None])
def test_after_signing_in_you_never_leave_the_site(client, target):
    assert sign_in(client, next=target or "").headers["location"] == "/"


def test_the_next_page_is_escaped_in_the_login_form(client):
    page = client.get('/login?next=/x"><script>alert(1)</script>').text
    assert "<script>alert(1)" not in page


def test_passwords_are_stored_salted_and_checked():
    one, two = auth.hash_password("pw"), auth.hash_password("pw")
    assert one != two and "pw" not in one
    assert auth.check_password("pw", one) and not auth.check_password("PW", one)


def test_a_web_page_sets_the_cookie_secure_over_https(app):
    with TestClient(app, base_url="https://testserver") as c:
        assert "secure" in sign_in(c).headers["set-cookie"].lower()


def test_users_come_from_the_config_and_the_variable():
    config = {"users": [{"username": "maria", "hash": "aa$bb"}, {"username": "luis", "hash": "cc$dd"}]}
    assert users_from({}) == {} and users_from(config) == {"maria": "aa$bb", "luis": "cc$dd"}
    assert users_from(config, "luis:new$hash, ana:ee$ff") == {"maria": "aa$bb", "luis": "new$hash", "ana": "ee$ff"}  # the variable wins


def test_the_password_script_creates_the_config_and_saves_or_replaces_a_user(tmp_path, monkeypatch, capsys):
    file = tmp_path / "config.json"
    monkeypatch.setattr(passwd, "CONFIG_FILE", str(file))
    answers = iter(["one", "one", "two", "two", "other", "other"])
    monkeypatch.setattr(passwd.getpass, "getpass", lambda prompt: next(answers))
    passwd.main(["maria"])
    config = json.loads(file.read_text())
    assert list(config) == ["users", "models"] and config["models"]["extraction_llm_model"] == {"model": "ollama_chat/gemma4:e2b", "api_key": "", "base_url": ""}
    assert set(config["models"]) == {"extraction_llm_model", "vlm_model", "embedding_model", "copilot_model", "judge_model"}
    config["models"]["judge_model"].update(model="openai/gpt-4o-mini", api_key="sk-test")  # the person edits the file in between
    file.write_text(json.dumps(config))
    passwd.main(["luis"])
    passwd.main(["maria"])  # a new password for maria replaces her entry
    config = json.loads(file.read_text())
    assert [u["username"] for u in config["users"]] == ["luis", "maria"] and list(config) == ["users", "models"]
    assert config["models"]["judge_model"]["api_key"] == "sk-test"  # nothing else was touched
    maria = config["users"][1]["hash"]
    assert auth.check_password("other", maria) and not auth.check_password("one", maria)
    assert f"USERS: maria:{maria}" in capsys.readouterr().out
    for bad in ("a:b", "a,b", ""):
        with pytest.raises(SystemExit):
            passwd.main([bad])
