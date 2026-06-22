# MNO Revenue Assurance — Agentic RAG Pipeline

An **agentic RAG** system that investigates **revenue leakage** for a mobile
network operator. A tool-using agent plans a multi-step investigation, runs
**SQL** against a DuckDB star-schema warehouse, retrieves **RA methodology** via
RAG, and **computes** the reconciliation — then writes a cited investigation
report.

Same A2P-bypass problem as Projects 1 & 2, now from the **finance/RA** side:
how much money is the operator losing, and where?

> Part 3 of 3 in the *Telecom RAG* series. See `otp-signature-analytics`
> (real-time RAG) and `otp-fraud-management` (multi-document RAG).

---

## The agent

```
            question
               │
   ┌───────────▼─────────────┐     tools (real, side-effecting reads)
   │  Agent                   │ ──► search_docs  (RAG over RA docs)
   │  • ReAct loop (Ollama)   │ ──► sql_query     (read-only DuckDB warehouse)
   │  • Scripted plan (offline)│ ──► compute       (safe arithmetic)
   └───────────┬─────────────┘
               │ observes results, iterates
               ▼
   cited investigation report (bypass leakage + billing gap + total)
```

### Two agent modes

| Mode | When | How |
|------|------|-----|
| **Tools + LLM writer** (default) | Ollama running | The agent deterministically orchestrates the tools (correct SQL, correct numbers) and the LLM *writes the report* from the verified findings. Robust even with a small model. |
| **Scripted** | No Ollama | Same tool orchestration; a templated report. Fully deterministic — what the tests use. |
| **LLM ReAct** (opt-in) | `TELCO_RA_REACT=1` + Ollama | The LLM plans and chooses tools/args itself. Recommended **only with a capable model** — small models (e.g. `llama3.2:3b`) author invalid SQL, so a guardrail blocks ungrounded answers and falls back to the scripted plan. |

> Design note: small local models are unreliable at authoring SQL in a free
> ReAct loop. The default therefore keeps tool orchestration deterministic and
> uses the LLM only for natural-language synthesis — fluent *and* correct.

**Two leakage sources** the agent quantifies:
1. **Bypass leakage** — OTPs delivered on non-revenue-bearing routes (OTT / SIM
   box / grey). `bypass_msgs × A2P_rate`.
2. **Billing/mediation gap** — delivered A2P messages finance never billed.
   `(delivered_a2p − billed_a2p) × A2P_rate`.

The **scripted plan** is the default and is fully deterministic (no LLM), so the
demo and tests always produce the same audited numbers. If Ollama is running,
the agent instead drives a generic **ReAct** loop, choosing tools itself.

---

## Star-schema warehouse (DuckDB + Parquet)

| Table | Grain | Notes |
|-------|-------|-------|
| `dim_enterprise` | enterprise | name, sender_id, sector |
| `dim_route` | route type | `is_revenue_bearing`, `a2p_rate_usd` |
| `fact_cdr` | one SMS record | raw source, `billed_usd` per message |
| `fact_otp_campaign` | enterprise × month × route | aggregated from `fact_cdr` |
| `fact_billing` | enterprise × month | finance's billed A2P (under-counts) |

ETL: raw CSV/JSON → `fact_cdr` → aggregate `fact_otp_campaign` → export Parquet.

---

## Quick start

```bash
cd mno-revenue-assurance
make setup
make pipeline                       # generate -> build warehouse -> index docs
make query SQL="SELECT route_type, SUM(message_count) FROM fact_otp_campaign GROUP BY 1"
make investigate Q="Estimate total A2P OTP revenue leakage and recommend controls."
make ui                             # Streamlit: warehouse overview + agent trace
make test                           # 6 tests, offline & deterministic
```

Runs with **no API keys and no Ollama**. Optional: `make setup-embeddings`
(semantic embeddings) and `./setup_ollama.sh` (LLM-driven ReAct + fluent report).

---

## Layout

```
ra_agent/
  config.py
  data/generate.py            synthetic CDRs + billing extract + RA docs
  pipeline/
    build_warehouse.py        ETL -> DuckDB star schema + Parquet
    index_docs.py             RA docs -> vector store
  agent/
    tools.py                  sql_query (read-only) · search_docs · compute
    agent.py                  scripted multi-step plan + LLM ReAct loop
  rag/                        embeddings, vector store, retriever, llm, fallback
  app/cli.py, app/streamlit_app.py
tests/
```

## Safety notes
- `sql_query` is **read-only**: it opens DuckDB in read-only mode and rejects any
  non-SELECT/WITH statement (INSERT/UPDATE/DELETE/DROP/… are blocked).
- `compute` uses an AST evaluator restricted to numbers and arithmetic operators
  — no function calls, no imports.

## Configuration (env vars)
Same as the other projects: `TELCO_RAG_FORCE_FALLBACK`, `TELCO_OLLAMA_MODEL`,
`OLLAMA_HOST`, `TELCO_EMBED_MODEL`. `AGENT_MAX_STEPS` lives in `config.py`.
