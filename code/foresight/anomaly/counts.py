"""
Daily ticket counts: the series every anomaly detector in Foresight
watches.

A ticket belongs to a supplier through the product it names. The
warehouse's `sku` column holds that product only when the body spells
the code in capitals, so the code is read again from the body, in any
case, and the column is used only as a fallback.

    tickets(con)                     one row per ticket, with supplier
    daily_counts(t, "supplier", ...) one column per supplier, one row a
                                     day, zeros where nothing was filed
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

SKU = re.compile(r"MRD-[A-Z]{3}-\d{3}")
DEFECT_SKU = "MRD-CLE-001"          # the Pemberton Mills cloth


def sku_in(body: str) -> str | None:
    """The first product code a ticket's body names, in any case."""
    m = SKU.search(body.upper())
    return m.group(0) if m else None


def tickets(warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Every ticket: its day, category, product and supplier."""
    with sqlite3.connect(warehouse) as con:
        t = pd.read_sql_query("""
            SELECT ticket_id, date(opened_at) AS day, category,
                   sku AS sku_column, body
            FROM tickets""", con)
        products = pd.read_sql_query("""
            SELECT p.sku, s.name AS supplier
            FROM products p JOIN suppliers s USING (supplier_id)""",
            con)
    t["day"] = pd.to_datetime(t.day)
    t["sku"] = t.body.map(sku_in).fillna(t.sku_column)
    t = t.merge(products, on="sku", how="left")
    return t.drop(columns="sku_column")


def daily_counts(t: pd.DataFrame, by: str, first: str,
                 last: str) -> pd.DataFrame:
    """Tickets per day and per value of `by`, every day from first to
    last, with a zero on days nothing was filed."""
    days = pd.date_range(first, last, freq="D", name="day")
    counts = (t[t.day.between(days[0], days[-1])]
              .groupby(["day", by]).size().unstack(fill_value=0))
    counts = counts.reindex(days, fill_value=0)
    counts.columns.name = by
    return counts.astype(int)


def all_series(t: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """Suppliers and categories side by side: the daily job's input.
    Columns are named 'supplier: X' and 'category: Y'."""
    parts = [daily_counts(t, by, first, last).add_prefix(f"{by}: ")
             for by in ("supplier", "category")]
    return pd.concat(parts, axis=1)
