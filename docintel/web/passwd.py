"""python -m docintel.web.passwd maria  ->  asks for the password and prints the 'maria:<hash>' entry for DOCINTEL_USERS
(several users: join the entries with commas; in a shell, put the whole value in single quotes because the hash has '$')."""
import getpass
import sys

from .auth import hash_password

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else input("Username: ").strip()
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat the password: "):
        raise SystemExit("The passwords differ.")
    print(f"{name}:{hash_password(password)}")
