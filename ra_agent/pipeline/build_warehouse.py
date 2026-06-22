"""ETL: raw CDRs/billing -> DuckDB star schema (+ Parquet export).

Star schema:
    dim_enterprise(enterprise_id, name, sender_id, sector)
    dim_route(route_type, is_revenue_bearing, a2p_rate_usd)
    fact_cdr(record_id, timestamp, month, enterprise_id, sender_id, route_type,
             is_bypass, billed_usd)
    fact_otp_campaign(enterprise_id, month, route_type, message_count, revenue_usd)
        -- aggregated from fact_cdr
    fact_billing(enterprise_id, month, billed_a2p_messages, billed_amount_usd)
"""
from __future__ import annotations

from typing import Optional

import duckdb

import config


def build(db_path: Optional[str] = None) -> dict:
    db_path = str(db_path or config.WAREHOUSE)
    con = duckdb.connect(db_path)

    cdr_csv = (config.RAW_DIR / "cdrs.csv").as_posix()
    bill_csv = (config.RAW_DIR / "billing.csv").as_posix()
    ent_json = (config.RAW_DIR / "enterprises.json").as_posix()

    # --- dimensions ---------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS dim_enterprise")
    con.execute(
        f"CREATE TABLE dim_enterprise AS SELECT * FROM read_json_auto('{ent_json}')"
    )

    con.execute("DROP TABLE IF EXISTS dim_route")
    con.execute(
        """
        CREATE TABLE dim_route AS
        SELECT * FROM (VALUES
            ('A2P_LICENSED', TRUE,  ?),
            ('OTT_WHATSAPP', FALSE, 0.0),
            ('OTT_TELEGRAM', FALSE, 0.0),
            ('SIM_BOX',      FALSE, 0.0),
            ('GREY_ROUTE',   FALSE, 0.0)
        ) AS t(route_type, is_revenue_bearing, a2p_rate_usd)
        """,
        [config.A2P_RATE_USD],
    )

    # --- facts --------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS fact_cdr")
    con.execute(
        f"CREATE TABLE fact_cdr AS SELECT * FROM read_csv_auto('{cdr_csv}', header=true)"
    )

    con.execute("DROP TABLE IF EXISTS fact_billing")
    con.execute(
        f"CREATE TABLE fact_billing AS SELECT * FROM read_csv_auto('{bill_csv}', header=true)"
    )

    con.execute("DROP TABLE IF EXISTS fact_otp_campaign")
    con.execute(
        """
        CREATE TABLE fact_otp_campaign AS
        SELECT enterprise_id, month, route_type,
               COUNT(*)         AS message_count,
               SUM(billed_usd)  AS revenue_usd
        FROM fact_cdr
        GROUP BY enterprise_id, month, route_type
        """
    )

    # --- export parquet -----------------------------------------------------
    counts = {}
    for tbl in ["dim_enterprise", "dim_route", "fact_cdr", "fact_billing", "fact_otp_campaign"]:
        out = (config.PARQUET_DIR / f"{tbl}.parquet").as_posix()
        con.execute(f"COPY {tbl} TO '{out}' (FORMAT PARQUET)")
        counts[tbl] = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

    con.close()
    return counts


if __name__ == "__main__":
    print(build())
