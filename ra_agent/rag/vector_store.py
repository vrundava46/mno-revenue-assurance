"""A small, persistent, dependency-light vector store.

Backed by numpy + JSON on disk. For the data sizes in these demos (hundreds to
low-thousands of chunks) an in-memory cosine search is instant and avoids the
install friction of a heavier engine. The interface mirrors the subset of
Chroma we use, so swapping in Chroma later is mechanical.

Persistence layout (one directory per collection):
    <dir>/<collection>.vectors.npy   float32 [N, D]
    <dir>/<collection>.meta.json     {ids, documents, metadatas}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Hit:
    id: str
    score: float
    document: str
    metadata: dict


class VectorStore:
    def __init__(self, directory: Path, collection: str = "default"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self._vec_path = self.dir / f"{collection}.vectors.npy"
        self._meta_path = self.dir / f"{collection}.meta.json"
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self._load()

    # --- persistence --------------------------------------------------------
    def _load(self) -> None:
        if self._vec_path.exists() and self._meta_path.exists():
            self.vectors = np.load(self._vec_path)
            meta = json.loads(self._meta_path.read_text())
            self.ids = meta["ids"]
            self.documents = meta["documents"]
            self.metadatas = meta["metadatas"]

    def persist(self) -> None:
        if self.vectors is None:
            self.vectors = np.zeros((0, 1), dtype=np.float32)
        np.save(self._vec_path, self.vectors)
        self._meta_path.write_text(
            json.dumps(
                {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}
            )
        )

    # --- writes -------------------------------------------------------------
    def has_id(self, _id: str) -> bool:
        return _id in set(self.ids)

    def existing_hashes(self) -> Dict[str, str]:
        """Map id -> content_hash for incremental indexing."""
        return {i: m.get("content_hash", "") for i, m in zip(self.ids, self.metadatas)}

    def add(self, ids, embeddings, documents, metadatas) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        if self.vectors is None or self.vectors.shape[0] == 0:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        """Add only ids not already present (used for incremental indexing)."""
        present = set(self.ids)
        keep = [j for j, i in enumerate(ids) if i not in present]
        if not keep:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.add(
            [ids[j] for j in keep],
            embeddings[keep],
            [documents[j] for j in keep],
            [metadatas[j] for j in keep],
        )

    # --- reads --------------------------------------------------------------
    def count(self) -> int:
        return len(self.ids)

    def query(self, embedding: np.ndarray, k: int = 5, where: Optional[dict] = None) -> List[Hit]:
        if self.count() == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        sims = self.vectors @ q  # vectors are L2-normalised -> cosine
        order = np.argsort(-sims)
        hits: List[Hit] = []
        for idx in order:
            meta = self.metadatas[idx]
            if where and any(meta.get(kk) != vv for kk, vv in where.items()):
                continue
            hits.append(Hit(self.ids[idx], float(sims[idx]), self.documents[idx], meta))
            if len(hits) >= k:
                break
        return hits
