"""Sign-in: a login page and a signed cookie. The app always asks for it: it will not start without users (DOCINTEL_USERS).

Users are "name:hash" pairs; make a hash with `python -m docintel.web.passwd`. The cookie holds "user|expiry|signature"
(HMAC with DOCINTEL_SECRET_KEY, or a random key per start, which signs everybody out when the app restarts)."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, RedirectResponse
from starlette.requests import Request

COOKIE = "docbase_session"
OPEN_PATHS = ("/login", "/static/")  # reachable without signing in
MAX_FAILURES, LOCK_SECONDS = 5, 300  # 5 wrong passwords for one user lock that user for 5 minutes


NO_USERS = ("DocBase asks everybody to sign in, and no users are configured.\n"
            "  1. python -m docintel.web.passwd maria        (asks for the password and prints maria:<hash>)\n"
            "  2. set DOCINTEL_USERS to that line (several users: join them with commas; in a shell use single quotes, the hash has '$')\n"
            "  3. start the app again")


def hash_password(password: str, salt: str | None = None) -> str:
    """'salt$hash' (scrypt)."""
    salt = salt or secrets.token_hex(16)
    return f"{salt}${hashlib.scrypt(password.encode(), salt=salt.encode(), n=2**14, r=8, p=1).hex()}"


def check_password(password: str, stored: str) -> bool:
    return hmac.compare_digest(hash_password(password, stored.split("$")[0]), stored)


_NOBODY = hash_password("", "0" * 32)  # checked when the user does not exist, so the answer takes as long either way


def go(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def safe_next(url) -> str:
    """Where to go after signing in: a path of this site, never another site."""
    url = str(url or "")
    return url if url.startswith("/") and not url.startswith("//") and "\\" not in url else "/"


class Auth:
    def __init__(self, users: dict[str, str], secret: str = "", hours: int = 8):
        self.users, self.hours = users, hours
        self._key = (secret or secrets.token_hex(32)).encode()
        self._failures: dict[str, deque] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return bool(self.users)

    def _sign(self, text: str) -> str:
        return hmac.new(self._key, text.encode(), "sha256").hexdigest()

    def token(self, user: str) -> str:
        text = f"{user}|{int(time.time()) + self.hours * 3600}"
        return f"{text}|{self._sign(text)}"

    def user_of(self, token: str | None) -> str | None:
        """The signed-in user a cookie value belongs to, or None (missing, forged, expired, user removed)."""
        text, _, signature = (token or "").rpartition("|")
        user, _, expires = text.rpartition("|")
        if not hmac.compare_digest(signature, self._sign(text)) or user not in self.users:
            return None
        return user if expires.isdigit() and int(expires) > time.time() else None

    def sign_in(self, user: str, password: str) -> tuple[str | None, str]:
        """(cookie value, "") when the password is right, otherwise (None, what to tell the person)."""
        if user not in self.users:
            check_password(password, _NOBODY)
            return None, "Wrong username or password."
        failures = self._failures[user]
        while failures and failures[0] < time.time() - LOCK_SECONDS:
            failures.popleft()
        if len(failures) >= MAX_FAILURES:
            return None, "Too many failed attempts. Wait a few minutes and try again."
        if check_password(password, self.users[user]):
            failures.clear()
            return self.token(user), ""
        failures.append(time.time())
        return None, "Wrong username or password."

    def deny(self, request: Request):
        """What someone who is not signed in gets: the login page for a page, 401 for the page's own requests (fetch, forms)."""
        if request.method in ("GET", "HEAD"):
            target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            return go(f"/login?next={quote(target, safe='')}")
        return PlainTextResponse("Sign in required.", status_code=401)


def auth_routes(site) -> APIRouter:
    router, auth = APIRouter(), site.auth

    @router.get("/login")
    def login_page(request: Request, next: str = "/"):
        if getattr(request.state, "user", None):
            return go(safe_next(next))
        return site.render(request, "login.html", active="", title="Sign in", next=safe_next(next), error="")

    @router.post("/login")
    async def login(request: Request):
        form = await request.form()
        token, error = auth.sign_in(str(form.get("username", "")).strip(), str(form.get("password", "")))
        if token is None:
            return site.render(request, "login.html", 401, active="", title="Sign in", next=safe_next(form.get("next")), error=error)
        response = go(safe_next(form.get("next")))
        response.set_cookie(COOKIE, token, max_age=auth.hours * 3600, httponly=True, samesite="lax", secure=request.url.scheme == "https")
        return response

    @router.post("/logout")
    def logout():
        response = go("/login")
        response.delete_cookie(COOKIE)
        return response

    return router
