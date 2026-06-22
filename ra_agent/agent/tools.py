"""Tools the revenue-assurance agent can call.

Each tool has a ``name``, a one-line ``description`` (shown to the LLM planner),
and a ``run(arg) -> str`` method returning a text observation. Tools are real:
SQL executes against the DuckDB warehouse, doc search hits the vector store, and
compute evaluates arithmetic safely.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import List, Optional

import duckdb

import config
from ra_agent.rag.retriever import Retriever


@dataclass
class ToolResult:
    observation: str
    sources: List[str]


# --------------------------------------------------------------------------
# SQL tool (read-only)
# --------------------------------------------------------------------------
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma)\b", re.I)


class SQLTool:
    name = "sql_query"
    description = (
        "Run a read-only SQL SELECT against the DuckDB warehouse. Tables: "
        "dim_enterprise(enterprise_id,name,sender_id,sector), "
        "dim_route(route_type,is_revenue_bearing,a2p_rate_usd), "
        "fact_cdr(record_id,timestamp,month,enterprise_id,sender_id,route_type,is_bypass,billed_usd), "
        "fact_otp_campaign(enterprise_id,month,route_type,message_count,revenue_usd), "
        "fact_billing(enterprise_id,month,billed_a2p_messages,billed_amount_usd)."
    )

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or config.WAREHOUSE)

    def _execute(self, sql: str, max_rows: int):
        s = sql.strip().rstrip(";")
        if _FORBIDDEN.search(s):
            raise ValueError("only read-only SELECT queries are allowed")
        if not re.match(r"(?is)^\s*(with|select)\b", s):
            raise ValueError("query must start with SELECT or WITH")
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            cur = con.execute(s)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows)
        finally:
            con.close()
        return cols, rows

    def query(self, sql: str, max_rows: int = 1000) -> List[dict]:
        """Structured access for internal use: returns a list of row dicts."""
        cols, rows = self._execute(sql, max_rows)
        return [dict(zip(cols, r)) for r in rows]

    def run(self, sql: str, max_rows: int = 50) -> ToolResult:
        try:
            cols, rows = self._execute(sql, max_rows)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"ERROR executing SQL: {e}", [])
        header = " | ".join(cols)
        body = "\n".join(" | ".join(f"{v}" for v in r) for r in rows)
        obs = f"{header}\n{body}" if rows else f"{header}\n(no rows)"
        return ToolResult(obs, sources=["warehouse.duckdb"])


# --------------------------------------------------------------------------
# Doc search tool (RAG)
# --------------------------------------------------------------------------
class DocSearchTool:
    name = "search_docs"
    description = (
        "Search the revenue-assurance methodology, controls and regulation docs "
        "for guidance (e.g. how to compute leakage). Input: a natural-language query."
    )

    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or Retriever()

    def run(self, query: str, k: int = 3) -> ToolResult:
        contexts = self.retriever.as_contexts(query, k=k)
        if not contexts:
            return ToolResult("No relevant methodology docs found.", [])
        obs = "\n".join(f"[{c['source']}] {c['text']}" for c in contexts)
        return ToolResult(obs, sources=[c["source"] for c in contexts])


# --------------------------------------------------------------------------
# Compute tool (safe arithmetic)
# --------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


class ComputeTool:
    name = "compute"
    description = "Evaluate an arithmetic expression, e.g. '12345 * 0.0065'. Numbers and + - * / ** % only."

    def run(self, expr: str) -> ToolResult:
        try:
            tree = ast.parse(expr, mode="eval")
            val = _safe_eval(tree.body)
            return ToolResult(f"{round(val, 4)}", sources=[])
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"ERROR: {e}", [])


def default_toolset() -> dict:
    return {t.name: t for t in (SQLTool(), DocSearchTool(), ComputeTool())}
