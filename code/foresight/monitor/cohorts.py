"""
The renewals as a deployed model meets them. Chapter 24 writes it.

The training table (data/build_table.py) holds only contracts whose
outcome is on record, because a model can learn only from those. The
monitor needs more: every contract that has reached its 90-day mark,
scored on the morning of the mark, whether or not anyone yet knows how
it ended. as_scored() builds those rows with Chapter 4's own functions,
so a contract appears here exactly as the table has it, and adds what
the monitor reads later: the notice date, the reason given with it,
and whether the account is a key account.

Three dates matter for every row, and known() and noticed() apply them:

    the mark        inputs and a score exist from this morning
    the notice      a leaver's notice arrives 60 days before the end,
                    30 days after the mark, with its reason
    the end date    the outcome is on record the day after

exposure() adds one column from the price_events table: the share of an
account's last year of spend on lines whose price Meridian raised
before the mark, for the accounts the rise applied to.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import (COLUMNS, MOMENT_DAYS, events,
                                        legacy_ids, segment_as_of,
                                        window_features)
from foresight.features.build import build as library
from foresight.features.definitions import STRATEGIC

GO_LIVE = pd.Timestamp("2025-01-01")
"""The day v0.6 was registered for production: the first morning on
which every validation outcome was on record."""
RECORD_ENDS = pd.Timestamp("2025-12-31")
"""The warehouse's last day. A notice is written only for contracts
that had ended by then."""
SUPPLIER_RISE = re.compile(r"^(.+) lines, non-key accounts$")


def due(con, first: str, last: str) -> pd.DataFrame:
    """Every contract ending in [first, last], outcome or not."""
    rows = pd.read_sql_query("""
        SELECT c.contract_id, c.account_id, c.end_date, c.term_months,
               c.legacy_terms, c.discount_pct, c.outcome,
               c.notice_date, c.cancellation_reason AS reason,
               a.is_key_account AS key_account
        FROM contracts c JOIN accounts a USING (account_id)
        WHERE c.end_date BETWEEN ? AND ?""", con,
        params=(first, last))
    rows["end_date"] = pd.to_datetime(rows.end_date)
    rows["notice_date"] = pd.to_datetime(rows.notice_date)
    rows["moment"] = rows.end_date - pd.Timedelta(days=MOMENT_DAYS)
    rows["discount_pct"] = rows.discount_pct.astype("Int64")
    known = rows.outcome.notna()
    rows["not_renewed"] = np.where(
        known, (rows.outcome == "not_renewed").astype(float), np.nan)
    return rows.drop(columns="outcome")


def as_scored(first: str, last: str, warehouse: Path = ML_WAREHOUSE,
              sources=None) -> pd.DataFrame:
    """Contracts ending in [first, last] as they stood at their marks:
    Chapter 4's columns, the library's supplier shares, the notice and
    the exposure to a price rise. `not_renewed` is NaN until known."""
    con = sqlite3.connect(warehouse)
    rows = due(con, first, last)
    orders, tickets = events(con, legacy_ids(con))
    accounts = pd.read_sql_query("""
        SELECT a.account_id, a.since, r.name AS region
        FROM accounts a JOIN regions r USING (region_id)""", con)
    accounts["since"] = pd.to_datetime(accounts.since)
    rises = price_events(con)
    table = (rows.merge(accounts, on="account_id", how="left")
                 .join(window_features(rows, orders, tickets),
                       on="contract_id")
                 .join(segment_as_of(rows, con), on="contract_id"))
    con.close()
    # The same finishing steps as build_table.build().
    table["tenure_days"] = (table.moment - table.since).dt.days
    counts = ["orders_90d", "orders_prev_90d", "tickets_90d"]
    table[counts] = table[counts].fillna(0).astype(int)
    table["spend_365"] = table.spend_365.fillna(0.0)
    table["days_since_order"] = table.days_since_order.astype("Int64")
    extra = ["notice_date", "reason", "key_account"]
    table = (table[COLUMNS + extra].sort_values(["moment",
                                                 "contract_id"])
                                    .reset_index(drop=True))
    shares = [f"supplier_{s.split()[0].lower()}" for s in STRATEGIC]
    enriched = library(table, shares, sources)
    table[shares] = enriched[shares].to_numpy()
    table["exposure"] = exposure(table, rises)
    return table


def price_events(con) -> pd.DataFrame:
    """The price_events table, with dates."""
    ev = pd.read_sql_query("SELECT * FROM price_events", con)
    ev["effective_on"] = pd.to_datetime(ev.effective_on)
    return ev


def exposure(rows: pd.DataFrame, rises: pd.DataFrame) -> np.ndarray:
    """The share of the year's spend on lines whose price rose before
    the mark, for long-tail accounts: 0 before a rise and for key
    accounts, the supplier's share after it."""
    out = np.zeros(len(rows))
    for r in rises.itertuples():
        m = SUPPLIER_RISE.match(r.scope)
        if m is None:
            continue                  # not one supplier's lines
        share = f"supplier_{m.group(1).split()[0].lower()}"
        after = (rows.moment > r.effective_on).to_numpy()
        tail = (rows.key_account == 0).to_numpy()
        out += np.where(after & tail, rows[share].to_numpy(), 0.0)
    return out


def known(rows: pd.DataFrame, day) -> pd.DataFrame:
    """Rows whose outcome was on record on `day`: ended before it."""
    return rows[(rows.end_date < pd.Timestamp(day))
                & rows.not_renewed.notna()]


def noticed(rows: pd.DataFrame, day) -> pd.Series:
    """True for the rows whose notice had arrived by `day`."""
    return rows.notice_date.notna() & (rows.notice_date
                                       < pd.Timestamp(day))


def notice_due(rows: pd.DataFrame) -> pd.Series:
    """The day by which a leaver's notice arrives: 60 days before the
    end, which is 30 days after the mark."""
    return rows.end_date - pd.Timedelta(days=60)
