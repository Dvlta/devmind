from __future__ import annotations

import hashlib
import json
import math
import re


DEFAULT_DIMENSION = 256

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

SYNONYMS = {
    "auth": ("authentication", "authenticate", "authorization", "login", "token", "session"),
    "authenticate": ("auth", "authentication", "login", "token"),
    "authentication": ("auth", "authenticate", "login", "token", "session"),
    "login": ("auth", "authenticate", "authentication", "session"),
    "issue": ("bug", "ticket", "problem", "failure", "error"),
    "bug": ("issue", "failure", "error", "problem"),
    "test": ("tests", "testing", "pytest", "unittest", "spec"),
    "tests": ("test", "testing", "pytest", "unittest", "spec"),
}


def embed_text(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    for token in expanded_tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def expanded_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(text):
        tokens.append(token)
        tokens.extend(SYNONYMS.get(token, ()))
    return tokens


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        raw = match.group(0)
        tokens.extend(split_identifier(raw))
    return [token for token in tokens if len(token) >= 2]


def split_identifier(value: str) -> list[str]:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", value.replace("_", " "))
    return [part.lower() for part in spaced.split() if part]


def serialize_vector(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def deserialize_vector(raw: str) -> list[float]:
    data = json.loads(raw)
    return [float(value) for value in data]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))

