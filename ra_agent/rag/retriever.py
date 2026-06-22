"""Dense retriever over the knowledge-base vector store."""
from __future__ import annotations

from typing import List

import config
from ra_agent.rag.embeddings import get_embedder
from ra_agent.rag.vector_store import Hit, VectorStore

KB_COLLECTION = "ra_docs"


def get_store() -> VectorStore:
    return VectorStore(config.VECTOR_DIR, KB_COLLECTION)


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or get_store()
        self.embedder = get_embedder()

    def search(self, query: str, k: int = 4) -> List[Hit]:
        q = self.embedder.encode(query)[0]
        return self.store.query(q, k=k)

    def as_contexts(self, query: str, k: int = 4) -> List[dict]:
        hits = self.search(query, k=k)
        return [
            {"text": h.document, "source": f"{h.metadata.get('source')}#{h.metadata.get('section')}", "score": h.score}
            for h in hits
        ]
