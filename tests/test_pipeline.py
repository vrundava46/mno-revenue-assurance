"""Tests for the agentic revenue-assurance pipeline (offline mode)."""
import pytest

import config
from ra_agent.agent.agent import Agent
from ra_agent.agent.tools import ComputeTool, SQLTool
from ra_agent.data import generate as datagen
from ra_agent.pipeline import build_warehouse, index_docs


@pytest.fixture(scope="module", autouse=True)
def built():
    datagen.generate_all(n_cdrs=6000, seed=7)
    build_warehouse.build()
    index_docs.index_docs()
    yield


# --- data + warehouse -------------------------------------------------------
def test_warehouse_tables_populated():
    sql = SQLTool()
    for tbl, _min in [("fact_cdr", 1), ("dim_enterprise", 8), ("dim_route", 5),
                      ("fact_otp_campaign", 1), ("fact_billing", 1)]:
        rows = sql.query(f"SELECT COUNT(*) AS c FROM {tbl}")
        assert rows[0]["c"] >= _min, tbl


def test_campaign_aggregation_matches_cdrs():
    sql = SQLTool()
    cdr = sql.query("SELECT COUNT(*) AS c FROM fact_cdr")[0]["c"]
    agg = sql.query("SELECT SUM(message_count) AS c FROM fact_otp_campaign")[0]["c"]
    assert cdr == agg


# --- tools ------------------------------------------------------------------
def test_sql_tool_blocks_writes():
    res = SQLTool().run("DELETE FROM fact_cdr")
    assert "ERROR" in res.observation


def test_compute_tool_math():
    assert ComputeTool().run("1000 * 0.0065").observation == "6.5"
    assert "ERROR" in ComputeTool().run("__import__('os')").observation


# --- agent ------------------------------------------------------------------
def test_agent_runs_multistep_investigation():
    res = Agent().run("Estimate total A2P OTP revenue leakage and recommend controls.")
    assert res.backend == "scripted"  # offline
    # multi-step: methodology, bypass sql, gap sql, compute, controls
    assert len(res.steps) >= 4
    actions = [s.action for s in res.steps]
    assert "search_docs" in actions and "sql_query" in actions and "compute" in actions
    assert "leakage" in res.answer.lower()
    assert "USD" in res.answer
    assert res.sources


def test_agent_quantifies_positive_leakage():
    res = Agent().run("How much revenue are we losing to OTP bypass?")
    # report should contain a positive total leakage figure
    import re
    nums = re.findall(r"USD ([\d.]+)", res.answer)
    assert nums and any(float(n) > 0 for n in nums)
