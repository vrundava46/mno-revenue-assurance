"""Markdown-aware chunking.

Splits documents on headings first, then packs paragraphs into ~chunk_chars
windows so each chunk is self-contained and small enough to embed well. Each
chunk keeps its source filename and the nearest heading as ``section``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str
    content_hash: str


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def chunk_markdown(text: str, source: str, chunk_chars: int = 700) -> List[Chunk]:
    section = "intro"
    buf: List[str] = []
    chunks: List[Chunk] = []

    def flush():
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            cid = f"{source}::{section}::{_hash(body)}"
            chunks.append(Chunk(cid, body, source, section, _hash(body)))
        buf.clear()

    cur_len = 0
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            cur_len = 0
            section = m.group(1).strip()[:80]
            continue
        buf.append(line)
        cur_len += len(line)
        if cur_len >= chunk_chars and line.strip() == "":
            flush()
            cur_len = 0
    flush()
    return chunks
