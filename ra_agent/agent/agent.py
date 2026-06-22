"""The revenue-assurance agent.

Two execution modes, same tools:

* **Scripted investigation** (default / offline): a deterministic, multi-step
  tool-using plan that recalls the methodology (search_docs), queries the
  warehouse for bypass leakage and billing-gap leakage (sql_query), reconciles
  the total (compute), and recalls controls (search_docs). Every step is real
  and recorded in the trace — no LLM required, fully reproducible.

* **ReAct loop** (when Ollama is reachable): the LLM chooses tools/arguments
  itself, observes results, and iterates until it emits a Final Answer.

Both return an ``AgentResult`` with the report, the step trace, and sources.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List

import config
from ra_agent.agent.tools import ComputeTool, DocSearchTool, SQLTool, default_toolset
from ra_agent.rag import llm

RATE = config.A2P_RATE_USD


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    backend: str = "scripted"


# --- SQL used by the scripted investigation --------------------------------
SQL_BYPASS = f"""
SELECT e.name AS enterprise,
       SUM(c.message_count) AS bypass_msgs,
       ROUND(SUM(c.message_count) * {RATE}, 2) AS bypass_leak_usd
FROM fact_otp_campaign c
JOIN dim_enterprise e USING (enterprise_id)
JOIN dim_route r USING (route_type)
WHERE r.is_revenue_bearing = FALSE
GROUP BY e.name
ORDER BY bypass_leak_usd DESC
"""

SQL_BILLING_GAP = f"""
WITH delivered AS (
    SELECT enterprise_id, month, SUM(message_count) AS delivered_a2p
    FROM fact_otp_campaign WHERE route_type = 'A2P_LICENSED'
    GROUP BY enterprise_id, month
)
SELECT SUM(d.delivered_a2p - b.billed_a2p_messages) AS gap_msgs,
       ROUND(SUM(d.delivered_a2p - b.billed_a2p_messages) * {RATE}, 2) AS gap_usd
FROM delivered d JOIN fact_billing b USING (enterprise_id, month)
"""


class Agent:
    def __init__(self):
        self.tools = default_toolset()
        self.sql: SQLTool = self.tools["sql_query"]
        self.docs: DocSearchTool = self.tools["search_docs"]
        self.calc: ComputeTool = self.tools["compute"]

    # ------------------------------------------------------------------
    def run(self, question: str) -> AgentResult:
        """Default: deterministic tool orchestration (correct, grounded numbers),
        with the LLM used to *write* the report when Ollama is available.

        Set TELCO_RA_REACT=1 to instead use the experimental LLM-planned ReAct
        loop (recommended only with a capable model — small models author bad SQL).
        """
        if os.environ.get("TELCO_RA_REACT") == "1" and llm.ollama_available():
            try:
                return self._react(question)
            except Exception:
                pass  # fall back to the grounded plan below

        steps, sources, f = self._investigate()

        if llm.ollama_available():
            try:
                report = self._llm_report(question, steps, f, _dedupe(sources))
                backend = f"tools+ollama:{config.OLLAMA_MODEL}"
            except Exception:
                report, backend = self._report(question, **f), "scripted"
        else:
            report, backend = self._report(question, **f), "scripted"

        return AgentResult(answer=report, steps=steps, sources=_dedupe(sources), backend=backend)

    # ------------------------------------------------------------------
    def _investigate(self):
        """Run the multi-step tool plan and return (steps, sources, findings)."""
        steps: List[AgentStep] = []
        sources: List[str] = []

        # 1. recall methodology
        m = self.docs.run("how to compute A2P OTP revenue leakage bypass and billing gap")
        steps.append(AgentStep("I need the RA methodology for quantifying leakage.",
                               "search_docs", "A2P OTP revenue leakage methodology", m.observation))
        sources += m.sources

        # 2. bypass leakage
        b = self.sql.run(SQL_BYPASS)
        steps.append(AgentStep("Quantify leakage from OTP traffic on non-revenue-bearing routes.",
                               "sql_query", "bypass leakage per enterprise", b.observation))
        sources += b.sources
        bypass_rows = self.sql.query(SQL_BYPASS)
        bypass_total = round(sum(r["bypass_leak_usd"] for r in bypass_rows), 2)
        top = bypass_rows[0]["enterprise"] if bypass_rows else "n/a"

        # 3. billing/mediation gap
        g = self.sql.run(SQL_BILLING_GAP)
        steps.append(AgentStep("Reconcile delivered A2P volume against what finance billed.",
                               "sql_query", "billing gap", g.observation))
        sources += g.sources
        gap_row = self.sql.query(SQL_BILLING_GAP)[0]
        gap_msgs = int(gap_row["gap_msgs"] or 0)
        gap_usd = float(gap_row["gap_usd"] or 0.0)

        # 4. reconcile total
        expr = f"{bypass_total} + {gap_usd}"
        c = self.calc.run(expr)
        steps.append(AgentStep("Total leakage = bypass leakage + billing gap.",
                               "compute", expr, c.observation))
        total = float(c.observation) if re.match(r"^-?\d", c.observation) else bypass_total + gap_usd

        # 5. recall controls
        ctl = self.docs.run("revenue assurance recommended controls for A2P bypass")
        steps.append(AgentStep("Recommend controls to remediate the leakage.",
                               "search_docs", "RA controls", ctl.observation))
        sources += ctl.sources

        findings = {"bypass_total": bypass_total, "top": top, "gap_msgs": gap_msgs,
                    "gap_usd": gap_usd, "total": round(total, 2)}
        return steps, sources, findings

    def _llm_report(self, question: str, steps: List[AgentStep], f: dict, sources: List[str]) -> str:
        """Use the LLM to synthesise the report from the VERIFIED findings.

        The LLM writes prose only — it does not author the SQL or the numbers,
        which were already computed deterministically. This keeps the output
        fluent AND correct even with a small local model.
        """
        from ra_agent.rag.llm import _ollama_generate  # type: ignore

        observations = "\n\n".join(
            f"[{s.action}] {s.action_input}\n{s.observation[:500]}" for s in steps
        )
        facts = (
            f"VERIFIED FIGURES (use these exact numbers, do not recompute):\n"
            f"- bypass_leakage_usd = {f['bypass_total']}\n"
            f"- largest_contributor = {f['top']}\n"
            f"- billing_gap_messages = {f['gap_msgs']}\n"
            f"- billing_gap_usd = {f['gap_usd']}\n"
            f"- total_leakage_usd = {f['total']}\n"
        )
        src_list = ", ".join(sources)
        prompt = (
            "You are a revenue-assurance analyst. Write a concise investigation "
            "report (markdown, with Findings / Method / Recommended controls "
            "sections) answering the question. Use ONLY the verified figures and "
            "the tool observations below. Do not invent numbers. Cite sources by "
            f"choosing from this exact list and putting them in [brackets]: {src_list}.\n\n"
            f"Question: {question}\n\n{facts}\n\nTOOL OBSERVATIONS:\n{observations}\n\nREPORT:"
        )
        return _ollama_generate(prompt)

    @staticmethod
    def _report(question, bypass_total, top, gap_msgs, gap_usd, total) -> str:
        return (
            f"# Revenue Assurance Investigation\n"
            f"_Question:_ {question}\n\n"
            f"## Findings\n"
            f"- **Bypass leakage:** USD {bypass_total} — OTP traffic delivered on "
            f"non-revenue-bearing routes (OTT/SIM-box/grey). Largest contributor: {top}.\n"
            f"- **Billing/mediation gap:** {gap_msgs} delivered A2P messages were "
            f"never billed = USD {gap_usd}.\n"
            f"- **Total estimated revenue leakage:** USD {round(total, 2)}.\n\n"
            f"## Method\n"
            f"Leakage = bypass_messages x A2P rate (USD {RATE}) + "
            f"(delivered_a2p - billed_a2p) x A2P rate, per RA methodology.\n\n"
            f"## Recommended controls\n"
            f"- Weekly CDR-vs-billing reconciliation; investigate gaps > 2%.\n"
            f"- Pin financial-sector sender IDs to licensed routes; enforce aggregator SLAs.\n"
            f"- Back-bill billing gaps where contractually permitted."
        )

    # ------------------------------------------------------------------
    # LLM ReAct loop (used only when Ollama is reachable)
    # ------------------------------------------------------------------
    def _react(self, question: str) -> AgentResult:
        from ra_agent.rag.llm import _ollama_generate  # type: ignore

        tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        instructions = (
            "You are a revenue-assurance agent. You MUST use tools to gather facts "
            "from the warehouse and docs before answering — never answer from memory.\n"
            f"Available tools:\n{tool_desc}\n\n"
            "Use EXACTLY this format. Emit ONE Action per turn and then STOP:\n"
            "Thought: <reasoning>\nAction: <one tool name>\nAction Input: <input>\n\n"
            "After you have used tools and seen Observations, finish with:\n"
            "Thought: <reasoning>\nFinal Answer: <the report with figures and sources>\n\n"
            "Example:\n"
            "Thought: I should recall how leakage is computed.\n"
            "Action: search_docs\nAction Input: how to compute A2P OTP revenue leakage\n\n"
            f"Question: {question}\n"
        )
        scratch = ""
        steps: List[AgentStep] = []
        sources: List[str] = []

        for _ in range(config.AGENT_MAX_STEPS):
            # stop before the model can hallucinate an Observation
            out = _ollama_generate(instructions + scratch, stop=["Observation:", "\nQuestion:"])

            has_final = "Final Answer:" in out
            action = _grab(out, "Action:")
            tool = self.tools.get(action)

            # Guardrail: do not accept a Final Answer until at least one tool ran.
            if has_final and steps:
                answer = out.split("Final Answer:", 1)[1].strip()
                return AgentResult(answer=answer, steps=steps, sources=_dedupe(sources),
                                   backend=f"react:{config.OLLAMA_MODEL}")
            if has_final and not steps and not tool:
                scratch += ("\nThought: (premature answer)\nObservation: You must call at "
                            "least one tool (start with search_docs or sql_query) before the Final Answer.")
                continue

            if not tool:
                scratch += f"\n{out.strip()}\nObservation: unknown tool '{action}'. Choose one of: {', '.join(self.tools)}."
                continue

            ainput = _grab(out, "Action Input:")
            thought = _grab(out, "Thought:")
            res = tool.run(ainput)
            sources += res.sources
            steps.append(AgentStep(thought, action, ainput, res.observation))
            scratch += (f"\nThought: {thought}\nAction: {action}\nAction Input: {ainput}\n"
                        f"Observation: {res.observation}\n")

        # ran out of steps, or the model never produced a grounded answer ->
        # fall back to the deterministic, fully-grounded plan.
        gsteps, gsources, f = self._investigate()
        return AgentResult(answer=self._report(question, **f), steps=gsteps,
                           sources=_dedupe(gsources),
                           backend=f"react:{config.OLLAMA_MODEL}->scripted-fallback")


def _grab(text: str, label: str) -> str:
    m = re.search(rf"{re.escape(label)}\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
