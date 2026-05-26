from __future__ import annotations

import re
import unicodedata


_RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE)
_RE_PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
_RE_NUM = re.compile(r"\b\d+\b")
_RE_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize SMS text for both classical ML and Transformer training.
    Keeps signal-bearing punctuation but stabilizes noisy entities.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")

    text = _RE_URL.sub(" <URL> ", text)
    text = _RE_EMAIL.sub(" <EMAIL> ", text)
    text = _RE_PHONE.sub(" <PHONE> ", text)
    text = _RE_NUM.sub(" <NUM> ", text)
    text = text.lower().strip()
    text = _RE_WS.sub(" ", text)
    return text

