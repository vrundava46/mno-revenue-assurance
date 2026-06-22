"""Streamlit UI: agentic RA investigations with a visible tool trace.

Run:  streamlit run ra_agent/app/streamlit_app.py   (from project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ra_agent.agent.agent import Agent  # noqa: E402
from ra_agent.agent.tools import SQLTool  # noqa: E402
from ra_agent.rag import llm  # noqa: E402

st.set_page_config(page_title="MNO Revenue Assurance", layout="wide")
st.title("💰 MNO Revenue Assurance — Agentic RAG")

backend = "Ollama (ReAct)" if llm.ollama_available() else "Scripted plan (offline)"
st.caption(f"Agent backend: **{backend}** · Tools: SQL over the DuckDB warehouse · RAG over RA methodology · compute.")

sql = SQLTool()
try:
    overview = sql.query(
        """
        SELECT r.is_revenue_bearing, SUM(c.message_count) AS msgs, ROUND(SUM(c.revenue_usd),2) AS revenue
        FROM fact_otp_campaign c JOIN dim_route r USING(route_type)
        GROUP BY r.is_revenue_bearing ORDER BY r.is_revenue_bearing DESC
        """
    )
except Exception:
    st.warning("Warehouse not built. Run `make pipeline` first.")
    st.stop()

c1, c2 = st.columns(2)
df = pd.DataFrame(overview)
c1.subheader("Revenue-bearing vs bypass volume")
c1.dataframe(df, use_container_width=True)

by_brand = sql.query(
    """
    SELECT e.name AS enterprise,
           ROUND(100.0 * SUM(CASE WHEN r.is_revenue_bearing THEN 0 ELSE c.message_count END)
                 / SUM(c.message_count), 1) AS bypass_pct
    FROM fact_otp_campaign c JOIN dim_enterprise e USING(enterprise_id)
    JOIN dim_route r USING(route_type)
    GROUP BY e.name ORDER BY bypass_pct DESC
    """
)
bb = pd.DataFrame(by_brand).set_index("enterprise")
c2.subheader("Bypass % by enterprise")
c2.bar_chart(bb)

st.subheader("🕵️ Run an investigation")
q = st.text_input("Investigation question", "Estimate total A2P OTP revenue leakage and recommend controls.")
if st.button("Run agent") and q:
    with st.spinner("Agent planning + executing tools..."):
        res = Agent().run(q)
    st.markdown(f"**Backend:** `{res.backend}` · **steps:** {len(res.steps)}")
    with st.expander("Tool trace", expanded=True):
        for i, s in enumerate(res.steps, 1):
            st.markdown(f"**Step {i} — `{s.action}`** · _{s.thought}_")
            st.code(s.observation[:600], language="text")
    st.markdown(res.answer)
    st.caption(f"Sources: {', '.join(res.sources)}")
