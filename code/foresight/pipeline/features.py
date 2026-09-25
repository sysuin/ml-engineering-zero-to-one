"""
One definition of Chapter 4's columns, for training and for serving.
Chapter 21 writes it.

build_table.build() makes rows only for contracts whose outcome is on
record, because it builds a training table and its SQL asks for the
outcome. The monthly list needs the same columns for contracts whose
outcome is not known yet: the ones reaching their mark today. Both
come from assemble() here, which calls build_table's own functions
(events(), legacy_ids(), window_features(), segment_as_of()), so every
column has one definition and a change to it reaches both paths.

    labelled()   the training table; test_pipeline.py checks that it
                 is Chapter 4's table to the last value
    at_mark()    the rows for one mark, whether the outcome is known
                 or not: what the monthly list is made from

skew() is the test that two ways of making rows agree: the same
contracts, compared column by column.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import (COLUMNS, MOMENT_DAYS, events,
                                        legacy_ids, renewals,
                                        segment_as_of, window_features)

INPUTS = [c for c in COLUMNS if c not in ("not_renewed",)]


class SkewError(Exception):
    """Two ways of making the same rows disagree."""


def assemble(con, rows: pd.DataFrame) -> pd.DataFrame:
    """Chapter 4's columns for `rows` (one per contract, with its
    moment): the tail of build_table.build(), in one place."""
    orders, tickets = events(con, legacy_ids(con))
    accounts = pd.read_sql_query("""
        SELECT a.account_id, a.since, r.name AS region
        FROM accounts a JOIN regions r USING (region_id)""", con)
    accounts["since"] = pd.to_datetime(accounts.since)
    table = (rows.merge(accounts, on="account_id", how="left")
                 .join(window_features(rows, orders, tickets),
                       on="contract_id")
                 .join(segment_as_of(rows, con), on="contract_id"))
    table["tenure_days"] = (table.moment - table.since).dt.days
    counts = ["orders_90d", "orders_prev_90d", "tickets_90d"]
    table[counts] = table[counts].fillna(0).astype(int)
    table["spend_365"] = table.spend_365.fillna(0.0)
    table["days_since_order"] = table.days_since_order.astype("Int64")
    keep = [c for c in COLUMNS if c in table.columns]
    if "not_renewed" in keep:
        table["not_renewed"] = table.not_renewed.astype(int)
    return (table[keep].sort_values(["moment", "contract_id"])
                       .reset_index(drop=True))


def labelled(first: str, last: str, known_by: str | None = None,
             warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """The training table for contracts ending from first to last."""
    with sqlite3.connect(warehouse) as con:
        return assemble(con, renewals(con, first, last, known_by))


def at_mark(mark, warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Every contract whose 90-day mark is `mark`, outcome or not."""
    return at_marks([mark], warehouse)


def at_marks(marks, warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """at_mark() for several marks at once."""
    ends = [str((pd.Timestamp(m) + pd.Timedelta(days=MOMENT_DAYS))
                .date()) for m in marks]
    with sqlite3.connect(warehouse) as con:
        rows = pd.read_sql_query(f"""
            SELECT contract_id, account_id, end_date, term_months,
                   legacy_terms, discount_pct
            FROM contracts
            WHERE end_date IN ({", ".join("?" * len(ends))})""",
            con, params=ends)
        rows["end_date"] = pd.to_datetime(rows.end_date)
        rows["moment"] = rows.end_date - pd.Timedelta(days=MOMENT_DAYS)
        rows["discount_pct"] = rows.discount_pct.astype("Int64")
        return assemble(con, rows)


def mark_on(day, warehouse: Path = ML_WAREHOUSE) -> pd.Timestamp:
    """The latest 90-day mark on or before `day`: the cohort whose
    list is made that morning."""
    with sqlite3.connect(warehouse) as con:
        (end,) = con.execute("""
            SELECT MAX(end_date) FROM contracts
            WHERE date(end_date, ?) <= ?""",
            (f"-{MOMENT_DAYS} days", str(pd.Timestamp(day).date()))
        ).fetchone()
    return pd.Timestamp(end) - pd.Timedelta(days=MOMENT_DAYS)


def skew(reference: pd.DataFrame, candidate: pd.DataFrame,
         columns=None, tolerance: float = 0.005) -> pd.DataFrame:
    """Compare two sets of rows for the same contracts, column by
    column: how many differ, and one contract that does. Numbers may
    differ by `tolerance`; a gap matches only a gap."""
    columns = [c for c in (columns or INPUTS)
               if c not in ("contract_id",)]
    a = reference.set_index("contract_id").sort_index()
    b = candidate.set_index("contract_id").reindex(a.index)
    out = []
    for c in columns:
        x, y = a[c], b[c]
        gap = x.isna() | y.isna()
        if pd.api.types.is_numeric_dtype(x.dtype):
            fx = x.astype(float).to_numpy()
            fy = y.astype(float).to_numpy()
            near = np.abs(fx - fy) <= tolerance
            same = np.where(gap, x.isna() & y.isna(), near)
        else:
            same = np.where(gap, x.isna() & y.isna(),
                            x.astype(str) == y.astype(str))
        differ = a.index[~np.asarray(same, bool)]
        out.append({"column": c, "rows": len(a),
                    "differ": len(differ),
                    "example": int(differ[0]) if len(differ) else None})
    return pd.DataFrame(out).set_index("column")


def check_skew(reference: pd.DataFrame, candidate: pd.DataFrame,
               columns=None, tolerance: float = 0.005) -> pd.DataFrame:
    """skew(), raising SkewError if any column differs anywhere."""
    found = skew(reference, candidate, columns, tolerance)
    bad = found[found.differ > 0]
    if len(bad):
        raise SkewError("; ".join(
            f"{c}: {r.differ} of {r.rows} rows differ (e.g. contract"
            f" {r.example})" for c, r in bad.iterrows()))
    return found
