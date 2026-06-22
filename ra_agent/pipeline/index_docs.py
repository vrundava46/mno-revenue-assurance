"""Index RA methodology/control/regulation docs into the vector store."""
from __future__ import annotations

from pathlib import Path
from typing import List

import config
from ra_agent.rag.chunking import Chunk, chunk_markdown
from ra_agent.rag.embeddings import get_embedder
from ra_agent.rag.retriever import get_store


def index_docs(docs_dir: Path | None = None) -> dict:
    docs_dir = docs_dir or config.DOCS_DIR
    chunks: List[Chunk] = []
    for md in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(md.read_text(), source=md.name))

    store = get_store()
    existing = set(store.existing_hashes().keys())
    new = [c for c in chunks if c.id not in existing]
    if not new:
        return {"indexed": 0, "total": store.count(), "skipped": len(chunks)}

    embedder = get_embedder()
    embeddings = embedder.encode([c.text for c in new])
    store.upsert(
        ids=[c.id for c in new],
        embeddings=embeddings,
        documents=[c.text for c in new],
        metadatas=[
            {"source": c.source, "section": c.section, "content_hash": c.content_hash}
            for c in new
        ],
    )
    store.persist()
    return {"indexed": len(new), "total": store.count(), "skipped": len(chunks) - len(new), "embedder": embedder.name}


if __name__ == "__main__":
    print(index_docs())
