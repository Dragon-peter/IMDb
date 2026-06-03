from __future__ import annotations

import html
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
TAG_PATTERN = re.compile(r"<[^>]+>")


def normalize_text(text: str) -> str:
    unescaped = html.unescape(text)
    stripped = TAG_PATTERN.sub(" ", unescaped)
    lowered = stripped.lower()
    compact = re.sub(r"\s+", " ", lowered).strip()
    return compact


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def english_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    ascii_letters = sum(1 for char in letters if char.isascii())
    return ascii_letters / len(letters)


def mostly_english(text: str, threshold: float = 0.7) -> bool:
    return english_ratio(text) >= threshold


def build_counter(tokenized_texts: list[list[str]]) -> Counter:
    counter: Counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    return counter
