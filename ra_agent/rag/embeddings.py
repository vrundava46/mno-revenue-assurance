"""Embedding backends with graceful degradation.

Preference order:
1. ``sentence-transformers`` (semantic, high quality) if installed & not forced off.
2. ``HashingEmbedder`` — a deterministic, dependency-free bag-of-words hashing
   vectorizer. Lower quality but always available, so the pipeline and tests
   run on any machine with just numpy.

Both backends expose the same ``encode(texts) -> np.ndarray`` interface and
return L2-normalised float32 vectors, so cosine similarity == dot product.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import List, Sequence

import numpy as np

import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic hashing vectorizer — no model download, no network."""

    name = "hashing-fallback"

    def __init__(self, dim: int = config.EMBED_DIM_FALLBACK):
        self.dim = dim

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
                # a bigram-ish second bucket sharpens the signal a little
                out[i, (h // self.dim) % self.dim] += 0.5
        return _normalize(out)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model to match the common interface."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vecs.astype(np.float32))


@lru_cache(maxsize=1)
def get_embedder():
    """Return the best available embedder (cached as a singleton)."""
    if not config.FORCE_FALLBACK:
        try:
            emb = SentenceTransformerEmbedder(config.EMBED_MODEL)
            return emb
        except Exception:
            # package missing, model download failed offline, etc. -> degrade
            pass
    return HashingEmbedder()
