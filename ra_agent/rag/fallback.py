"""Deterministic, offline answer synthesiser.

Used whenever no LLM (Ollama) is reachable or when ``TELCO_RAG_FORCE_FALLBACK``
is set. It performs lightweight *extractive* answering: it selects the most
relevant sentences from the retrieved context and stitches them into a cited
answer. No network, no model — just string processing — so it always works and
is fully deterministic (great for tests).
"""
from __future__ import annotations

import re
from typing import List

_WORD = re.compile(r"[a-z0-9]+")


def _keywords(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 0]


def extractive_answer(question: str, contexts: List[dict], max_sentences: int = 4) -> str:
    """contexts: list of {"text": str, "source": str}. Returns a cited answer."""
    if not contexts:
        return "I don't have any indexed information to answer that yet."

    q_kw = _keywords(question)
    scored = []
    for ci, ctx in enumerate(contexts):
        src = ctx.get("source", f"doc{ci}")
        for sent in _split_sentences(ctx["text"]):
            overlap = len(q_kw & _keywords(sent))
            if overlap == 0:
                continue
            scored.append((overlap, len(sent), sent, src))

    if not scored:
        # nothing matched the question words: fall back to the top context lead
        lead = _split_sentences(contexts[0]["text"])[:2]
        src = contexts[0].get("source", "doc0")
        body = " ".join(lead)
        return f"{body} [source: {src}]"

    scored.sort(key=lambda t: (-t[0], t[1]))
    seen, chosen = set(), []
    for _, _, sent, src in scored:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sent, src))
        if len(chosen) >= max_sentences:
            break

    lines = [f"{sent} [source: {src}]" for sent, src in chosen]
    return "Based on the indexed knowledge base:\n" + "\n".join(f"- {l}" for l in lines)
