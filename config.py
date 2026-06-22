"""Central configuration for MNO Revenue Assurance (Agentic RAG)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_store"
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = DATA_DIR / "docs"          # RA methodology / control / regulation docs
PARQUET_DIR = DATA_DIR / "parquet"    # exported star-schema tables
WAREHOUSE = DATA_DIR / "warehouse.duckdb"
VECTOR_DIR = DATA_DIR / "vectors"

for _d in (DATA_DIR, RAW_DIR, DOCS_DIR, PARQUET_DIR, VECTOR_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.environ.get("TELCO_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM_FALLBACK = 256

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("TELCO_OLLAMA_MODEL", "llama3.2:3b")

FORCE_FALLBACK = os.environ.get("TELCO_RAG_FORCE_FALLBACK", "0") == "1"

# A2P termination fee the MNO earns per correctly-routed enterprise OTP.
A2P_RATE_USD = 0.0065
ROUTE_TYPES = ["A2P_LICENSED", "OTT_WHATSAPP", "OTT_TELEGRAM", "SIM_BOX", "GREY_ROUTE"]
BYPASS_ROUTES = {"OTT_WHATSAPP", "OTT_TELEGRAM", "SIM_BOX", "GREY_ROUTE"}

# Max ReAct iterations before the agent must answer.
AGENT_MAX_STEPS = 6
