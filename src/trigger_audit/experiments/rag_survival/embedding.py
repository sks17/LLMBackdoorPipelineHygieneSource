"""Deterministic hash embedding for the RAG delivery baseline (controlled, reproducible ranking).

Not a semantic model: it maps text to a hashed bag-of-words vector (L2-normalized), so retrieval
ranking is decided by lexical overlap and is identical across runs. A plumbing/delivery check needs
that determinism -- a real embedding's semantic quirks would make the ranking (and therefore the
"excluded" condition) flaky. Implements LangChain's ``Embeddings`` interface so it drops into a
real ``InMemoryVectorStore``.
"""

from __future__ import annotations

import math
import zlib

from langchain_core.embeddings import Embeddings


class DeterministicHashEmbedding(Embeddings):
    """A dependency-free, deterministic bag-of-words embedding (hashed tokens, L2-normalized)."""

    embedding_id = "deterministic-hash-256"

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def _embed(self, text: str) -> list[float]:
        """Hash each alphanumeric token into a fixed-width vector and L2-normalize it."""
        vector = [0.0] * self._dim
        for raw_token in text.lower().split():
            token = "".join(ch for ch in raw_token if ch.isalnum())
            if token:
                vector[zlib.crc32(token.encode("utf-8")) % self._dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
