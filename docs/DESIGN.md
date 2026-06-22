# Telecom RAG Projects — Design Spec

**Date:** 2026-06-20
**Author:** generated with Claude Code
**Status:** Approved (design), implementation in progress

## 1. The Unifying Domain Problem — A2P OTP Bypass / OTT Grey-Routing

Enterprises (banks, fintech, e-commerce) send **One-Time Passwords (OTPs)** to their
customers. Legitimately these are **Application-to-Person (A2P) SMS** and must traverse
**licensed A2P routes**, on which the **Mobile Network Operator (MNO)** earns
termination / A2P messaging revenue.

Third-party aggregators and "grey-route" vendors **bypass** the licensed A2P channel to
keep the margin for themselves and starve the MNO of revenue. Common bypass techniques:

- **OTT delivery** — sending the OTP over Over-The-Top apps (WhatsApp Business API,
  Telegram, etc.). The MNO carries no SMS and earns nothing.
- **SIM farms / SIM boxes** — injecting A2P traffic as cheap **P2P** SMS from racks of
  SIMs to dodge A2P pricing.
- **Grey routes** — international rerouting and sender-ID spoofing to disguise traffic.

The business impact is **revenue leakage** for the MNO plus **fraud / deliverability /
security** risk for the enterprise and subscriber.

The three projects attack this single problem from three angles, each demonstrating a
different RAG architecture.

## 2. The Three Projects

| # | Folder | RAG Architecture | Angle on the problem |
|---|--------|------------------|----------------------|
| 1 | `otp-signature-analytics` | Real-Time Assistant RAG | Fingerprint the **live OTP stream**, flag traffic likely **bypassing A2P** (OTT / grey route / SIM-box) in real time; assistant answers operator questions grounded in a routing/regulation KB. |
| 2 | `otp-fraud-management` | Multi-Document RAG | Help a fraud analyst **investigate** bypass/fraud across heterogeneous docs (threat-intel, SOPs, regulations, case files) with hybrid retrieval + cross-document citations. |
| 3 | `mno-revenue-assurance` | Agentic RAG Pipeline | A tool-using agent **quantifies & reconciles revenue leakage** from bypassed OTP volume across a CDR/billing/A2P warehouse + RA methodology docs. |

Each project is **fully standalone** (own `requirements.txt`, README, data, tests).

## 3. Shared Technical Foundation

- **Language:** Python 3.11
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) — fully local, offline after first download.
- **Vector store:** ChromaDB (local persistent, no Docker).
- **Analytics / warehouse:** DuckDB + Parquet (no Docker).
- **Sparse retrieval:** `rank_bm25` (for the hybrid retriever in P2).
- **LLM:** **Ollama** (`llama3.2:3b`), via `setup_ollama.sh`. Every project includes a
  **deterministic offline "extractive" answerer** so the full pipeline and the test
  suite pass with **zero API keys and no Ollama** — the LLM is an enhancement, never a
  hard dependency. This is what guarantees "it works on my PC".
- **Synthetic data:** `Faker` + seeded RNG generators. No external data is ever required.
- **UI:** CLI (`typer`) for scripting/tests + **Streamlit** chat UI.
- **Tests:** `pytest`, CI-safe (offline mode forced).

### Common internal layout (per project)
```
<project>/
  README.md
  requirements.txt
  Makefile
  setup_ollama.sh
  config.py
  src/<pkg>/
    data/        # synthetic generators
    pipeline/    # ETL / streaming / indexing
    rag/         # embeddings, vector store, retriever, llm, fallback
    app/         # cli.py, streamlit_app.py
  tests/
```

### LLM abstraction (shared pattern)
`rag/llm.py` exposes `answer(question, context_chunks) -> Answer`. It tries Ollama
(`/api/generate`); on any failure it falls back to `rag/fallback.py`, a deterministic
extractive summarizer that stitches the top retrieved chunks into a cited answer. A
`TELCO_RAG_FORCE_FALLBACK=1` env var forces fallback (used by tests).

## 4. Project 1 — OTP Signature Analytics (Real-Time Assistant RAG)

**Data engineering**
- `data/generate.py` — synthetic A2P SMS event generator: `sender_id`, `message_text`,
  `brand`, `country`, `route_type` (A2P_LICENSED / OTT_WHATSAPP / SIM_BOX / GREY_ROUTE),
  `is_otp`, `delivered`, `latency_ms`, `cost_to_mno`, `timestamp`.
- KB docs (Markdown): A2P routing policy, sender-ID registry, OTP template/DLT rules,
  bypass-detection playbook.
- `pipeline/stream_simulator.py` — producer writing events to a local queue (file/JSONL
  tail or `queue.Queue`); `pipeline/realtime.py` consumer that classifies each event
  (OTP? bypass-suspect?), extracts the **signature** (template skeleton + sender pattern),
  and updates rolling aggregates persisted to DuckDB.
- `pipeline/index_kb.py` — chunk + embed KB docs into Chroma, **incremental** (skip
  unchanged by content hash).

**RAG / assistant**
- Retrieves KB context **and** live traffic stats from DuckDB, answers operator
  questions, e.g. *"Which brands show OTP bypass on OTT this hour and what's the policy?"*

**Signature extraction:** normalize digits→`#`, mask codes, derive a template hash; group
by `(sender_id, template_hash)` to define a "signature"; anomaly = signature seen on a
non-A2P route or with abnormal volume/latency.

## 5. Project 2 — OTP Fraud Management (Multi-Document RAG)

**Data engineering**
- `data/generate.py` — heterogeneous corpus:
  - Markdown + generated **PDF** policy/SOP/threat-intel docs (fraud typologies: OTT
    bypass, SIM-box, grey route, OTP interception, smishing).
  - `cases.csv` historical investigation cases; `iocs.json` indicators of compromise.
- `pipeline/ingest.py` — loaders per type (PDF via `pypdf`, MD, CSV, JSON), normalize,
  chunk with metadata (`source`, `doc_type`, `section`, `case_id`), write a **catalog**
  table in DuckDB; embed into Chroma; **incremental** by content hash.

**RAG**
- **Hybrid retrieval:** BM25 (`rank_bm25`) + dense (Chroma), score fusion (RRF), optional
  cross-encoder-free re-rank by keyword overlap.
- Answers include **cross-document citations** `[source#section]`.

## 6. Project 3 — MNO Revenue Assurance (Agentic RAG Pipeline)

**Data engineering**
- `data/generate.py` — synthetic **CDRs** (call/SMS detail records), **billing** records,
  **rate plans**, **interconnect/A2P agreements**, and a stream of **enterprise OTP
  campaigns** with an expected-vs-actual A2P delivery split (the leakage).
- `pipeline/build_warehouse.py` — ETL into a **DuckDB star schema**
  (`fact_cdr`, `fact_billing`, `dim_rate_plan`, `dim_route`, `dim_enterprise`) + Parquet.
- `pipeline/index_docs.py` — RA methodology / control / regulatory docs into Chroma.

**Agent**
- ReAct-style loop (`agent/agent.py`) with tools:
  - `sql_query(sql)` — read-only DuckDB over the warehouse.
  - `search_docs(query)` — RAG over RA docs.
  - `compute(expression)` — safe arithmetic for reconciliation.
- Plan → act → observe → reflect; produces a structured **investigation report**
  (leakage estimate + cited methodology + recommended controls).
- Works with Ollama for planning; includes a **scripted deterministic planner fallback**
  for canned investigation types so the demo + tests run offline.

## 7. Testing & "it works" guarantee

- Each project: `make setup && make data && make pipeline && make test`.
- Tests force `TELCO_RAG_FORCE_FALLBACK=1` → no Ollama/network needed.
- Tests cover: data generation shape, ETL/warehouse row counts, signature extraction,
  retriever returns relevant chunk, hybrid fusion ordering, agent tool execution, and an
  end-to-end "ask a question, get a cited answer" smoke test.
- README documents the optional Ollama upgrade path for nicer natural-language answers.

## 8. Out of scope (YAGNI)

- No Docker/Kubernetes, no cloud, no Kafka (a local streaming simulator stands in).
- No auth, no real PII, no real carrier integrations.
- Embedding/LLM model fine-tuning.
