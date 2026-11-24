"""python -m docintel.web.passwd maria  ->  asks for the password and saves maria (name and password hash) under "users" in config.json,
replacing her entry if she already exists, so it also changes a password. Everything else in the file is left as it is; a missing file
is created with the default models, ready to edit. The "name:hash" line is printed too: servers can put it in USERS
(several users: join the lines with commas; in a shell use single quotes, the hash has '$')."""
import getpass
import json
import sys
from pathlib import Path

from ..shared.settings import CONFIG_FILE, DEFAULT_MODELS, load_config
from .auth import hash_password


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    name = (args[0] if args else input("Username: ")).strip()
    if not name or any(c in name for c in ":,|\n"):
        raise SystemExit("The username cannot be empty or contain : , or |")
    password = getpass.getpass("Password: ")
    if not password or password != getpass.getpass("Repeat the password: "):
        raise SystemExit("The password is empty or the two do not match.")
    password_hash = hash_password(password)
    config = load_config(CONFIG_FILE) or {"models": {key: {"model": model, "api_key": "", "base_url": ""} for key, model in DEFAULT_MODELS.items()}}
    users = [u for u in config.get("users", []) if u.get("username") != name] + [{"username": name, "hash": password_hash}]
    file = Path(CONFIG_FILE)
    file.write_text(json.dumps({"users": users, **{k: v for k, v in config.items() if k != "users"}}, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        file.chmod(0o600)  # it may hold API keys: only its owner reads it (no effect on Windows)
    except OSError:
        pass
    print(f"Saved {name} in {file.resolve()}\nFor USERS: {name}:{password_hash}")


if __name__ == "__main__":
    main()
