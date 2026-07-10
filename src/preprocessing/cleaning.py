"""Aggressive text cleaning for the classical ML pipeline.

Kept in src/ (not a notebook) so training and the API clean text identically.
"""
import re

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\.\S+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation and collapse whitespace.

    Order matters: lowercase -> HTML -> URLs -> non-letters -> whitespace.
    """
    text = text.lower()
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text
