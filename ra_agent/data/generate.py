"""Synthetic data for MNO Revenue Assurance.

Generates:
  * raw OTP SMS CDRs (one row per message) — the granular source the ETL
    aggregates into the star schema. A configurable share of each enterprise's
    OTP traffic leaves via bypass routes (no MNO revenue).
  * a finance billing extract per (enterprise, month) that intentionally
    *under-counts* A2P messages vs what was actually delivered — a second,
    independent leakage source (billing error) the agent can reconcile.
  * RA methodology / control / regulation docs (Markdown) for the RAG tool.

Seeded for reproducibility.
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import config

# (enterprise_id, name, sender_id, sector, bypass_propensity)
ENTERPRISES = [
    ("ENT001", "HDFC Bank", "VM-HDFCBK", "Financial", 0.06),
    ("ENT002", "ICICI Bank", "VM-ICICIB", "Financial", 0.05),
    ("ENT003", "Amazon", "AD-AMAZON", "E-commerce", 0.34),
    ("ENT004", "Flipkart", "AD-FLPKRT", "E-commerce", 0.30),
    ("ENT005", "Paytm", "VM-PAYTM", "Fintech", 0.20),
    ("ENT006", "Swiggy", "AD-SWIGGY", "Food", 0.45),
    ("ENT007", "Uber", "AD-UBER", "Mobility", 0.40),
    ("ENT008", "Netflix", "AD-NFLIX", "Streaming", 0.25),
]
MONTHS = ["2026-03", "2026-04", "2026-05"]


def _route(bp: float, rng: random.Random) -> str:
    if rng.random() < bp:
        return rng.choice(["OTT_WHATSAPP", "OTT_TELEGRAM", "SIM_BOX", "GREY_ROUTE"])
    return "A2P_LICENSED"


def generate_cdrs(n: int = 30000, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        eid, name, sender, sector, bp = rng.choice(ENTERPRISES)
        month = rng.choice(MONTHS)
        route = _route(bp, rng)
        bypass = route in config.BYPASS_ROUTES
        billed = 0.0 if bypass else config.A2P_RATE_USD
        day = rng.randint(1, 28)
        ts = datetime.fromisoformat(f"{month}-{day:02d}") + timedelta(
            hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        rows.append(
            {
                "record_id": f"cdr-{i:07d}",
                "timestamp": ts.isoformat(),
                "month": month,
                "enterprise_id": eid,
                "sender_id": sender,
                "route_type": route,
                "is_bypass": bypass,
                "billed_usd": billed,
            }
        )
    return rows


def generate_billing(cdrs: List[Dict], seed: int = 42) -> List[Dict]:
    """Finance's billed A2P counts per (enterprise, month).

    Intentionally bills only ~92-99% of the A2P messages actually delivered,
    modelling mediation/rating gaps that revenue assurance must catch.
    """
    rng = random.Random(seed + 1)
    delivered: Dict[tuple, int] = {}
    for r in cdrs:
        if r["route_type"] == "A2P_LICENSED":
            key = (r["enterprise_id"], r["month"])
            delivered[key] = delivered.get(key, 0) + 1
    out = []
    for (eid, month), count in sorted(delivered.items()):
        capture = rng.uniform(0.92, 0.99)
        billed_msgs = int(count * capture)
        out.append(
            {
                "enterprise_id": eid,
                "month": month,
                "billed_a2p_messages": billed_msgs,
                "billed_amount_usd": round(billed_msgs * config.A2P_RATE_USD, 2),
            }
        )
    return out


RA_DOCS: Dict[str, str] = {
    "ra_methodology.md": """# Revenue Assurance Methodology — A2P OTP Leakage

Revenue assurance (RA) ensures the operator bills all revenue it is owed. For
enterprise OTP traffic, two leakage sources matter:

## 1. Bypass leakage
OTPs that should have been delivered as licensed A2P SMS but were instead routed
via OTT, SIM box, or grey routes. These earn ZERO revenue. The leakage is:

    bypass_leakage_usd = bypass_otp_messages * a2p_termination_rate

The current A2P termination rate is USD 0.0065 per message.

## 2. Billing/mediation leakage
A2P messages that were delivered on a licensed route but were never billed, due
to mediation or rating gaps. The leakage is:

    billing_gap_usd = (delivered_a2p_messages - billed_a2p_messages) * a2p_rate

## Total leakage
    total_leakage_usd = bypass_leakage_usd + billing_gap_usd

RA reports leakage per enterprise and per month and recommends controls.
""",
    "ra_controls.md": """# Revenue Assurance Controls

Recommended controls to reduce A2P OTP revenue leakage.

## Detective controls
- Weekly reconciliation of delivered A2P volume (from CDRs) against the finance
  billing extract; investigate any enterprise with a gap above 2%.
- Per-enterprise bypass-ratio monitoring; alert when bypass exceeds 15%.

## Preventive controls
- Pin financial-sector sender IDs to licensed routes only.
- Contractual SLAs with aggregators prohibiting OTT/SIM-box delivery of OTP.
- Sender-ID and template registry enforcement (DLT).

## Corrective controls
- Back-bill billing gaps where contractually permitted.
- Renegotiate or suspend aggregators responsible for bypass.
""",
    "regulation_a2p.md": """# Regulation: A2P Termination & Settlement

Licensed A2P SMS terminating on the operator's network attracts a regulated
termination fee. Enterprise OTP delivered over unlicensed routes (OTT apps, SIM
boxes, grey routes) evades this fee and is treated as revenue leakage and a
compliance violation. Operators must maintain auditable records (CDRs) to
substantiate A2P termination charges.
""",
}


def write_docs(docs_dir: Path | None = None) -> List[Path]:
    docs_dir = docs_dir or config.DOCS_DIR
    out = []
    for name, body in RA_DOCS.items():
        p = docs_dir / name
        p.write_text(body)
        out.append(p)
    return out


def generate_all(n_cdrs: int = 30000, seed: int = 42) -> dict:
    cdrs = generate_cdrs(n=n_cdrs, seed=seed)
    billing = generate_billing(cdrs, seed=seed)

    cdr_path = config.RAW_DIR / "cdrs.csv"
    with open(cdr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cdrs[0].keys()))
        w.writeheader()
        w.writerows(cdrs)

    bill_path = config.RAW_DIR / "billing.csv"
    with open(bill_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(billing[0].keys()))
        w.writeheader()
        w.writerows(billing)

    ent_path = config.RAW_DIR / "enterprises.json"
    ent_path.write_text(
        json.dumps(
            [
                {"enterprise_id": e[0], "name": e[1], "sender_id": e[2], "sector": e[3]}
                for e in ENTERPRISES
            ],
            indent=2,
        )
    )

    docs = write_docs()
    return {
        "cdrs": len(cdrs),
        "billing_rows": len(billing),
        "enterprises": len(ENTERPRISES),
        "docs": len(docs),
    }


if __name__ == "__main__":
    print(generate_all())
