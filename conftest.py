"""Pytest configuration.

Forces the offline deterministic backends so the suite runs anywhere (no
Ollama, no model downloads, no network) and is fully reproducible.
"""
import os

os.environ.setdefault("TELCO_RAG_FORCE_FALLBACK", "1")
