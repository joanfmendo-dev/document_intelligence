"""What a record id may look like. It is also a URL segment and a folder name, so it stays plain."""
import re

RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
