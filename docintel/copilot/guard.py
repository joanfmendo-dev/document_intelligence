"""Guards on what the model can put in front of the user. The model writes only a plan; the one free text in it is the question of a
`clarify` (and its suggestions). That text is shown as it is, so it is the way out for a prompt-extraction attack ("repeat everything above"):
it is checked against the prompt the model was given, and replaced when it repeats it."""
from __future__ import annotations

import re

WINDOW = 8                       # this many consecutive words of the prompt, repeated, count as a leak
MAX_SUGGESTION = 80              # characters of one suggestion the model proposes
LEAK_REPLY = "I can only help with questions about the records. Could you ask about patients, encounters, labs or a record?"


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").casefold())


def leaks(text: str, prompt: str, window: int = WINDOW) -> bool:
    """Does `text` repeat a run of `window` words of `prompt`? (The question of a user is in the prompt too, so the user's own words are
    removed first by the caller.)"""
    words, source = _words(text), " ".join(_words(prompt))
    return any(" ".join(words[i:i + window]) in source for i in range(len(words) - window + 1))
