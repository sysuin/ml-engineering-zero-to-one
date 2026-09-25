"""
The daily check on the order feed: what Meridian paid its suppliers,
and what its long-tail accounts paid Meridian, for each supplier's
lines, against what they usually pay. Chapter 24 writes it.

Chapter 18 said that for prices the right detector is often the rule
itself. The rule here is the catalogue: every line has a list price and
a catalogue cost, and an account's agreed discount. Two indexes per
supplier and segment per day, over long-tail lines:

    price    what the account paid / (list price less its discount)
    cost     what Meridian paid / the catalogue cost

Each is compared with its own median over the 56 days before (Chapter
18's window). A move of SHIFT or more, on a day with at least LINES
lines, is an alert. After an alert the series is quiet for a window,
while its baseline fills with the new level: a price that has changed
is news once, not every day for eight weeks. The threshold is half the
smallest price change Meridian has made, the 6% small-business rise of
July 2023.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

WINDOW = 56
SHIFT = 0.03
LINES = 5


def daily(warehouse: Path = ML_WAREHOUSE,
          segment: str = "on the day") -> pd.DataFrame:
    """Price and cost indexes by day, for each supplier's lines to
    each segment of the long tail ("Voss Industrial / Mid-market").
    The segment is the account's on the day of the order, from the
    history; segment="today" reads today's, which is Chapter 13's
    mistake and is kept to show what it costs."""
    seg = ("a.segment" if segment == "today"
           else "COALESCE(h.segment, a.segment)")
    with sqlite3.connect(warehouse) as con:
        d = pd.read_sql_query(f"""
            SELECT o.order_date AS day,
                   s.name || ' / ' || {seg} AS supplier,
                   COUNT(*) AS lines,
                   AVG(l.unit_price / (p.list_price
                       * (1 - l.discount_pct / 100.0))) AS price,
                   AVG(l.unit_cost / p.unit_cost) AS cost
            FROM order_lines l
            JOIN orders o USING (order_id)
            JOIN products p USING (sku)
            JOIN suppliers s USING (supplier_id)
            JOIN accounts a ON a.account_id = o.account_id
            LEFT JOIN account_history h
              ON h.account_id = o.account_id
             AND o.order_date >= h.valid_from
             AND o.order_date <= COALESCE(h.valid_to, '9999-12-31')
            WHERE a.is_key_account = 0
            GROUP BY 1, 2""", con)
    d["day"] = pd.to_datetime(d.day)
    return d


def check(d: pd.DataFrame, what: str, shift: float = SHIFT,
          window: int = WINDOW, least: int = LINES) -> pd.DataFrame:
    """One row per alert: a day on which a series' index moved `shift`
    or more from its trailing median, and had not alerted in the
    `window` days before."""
    x = d.pivot(index="day", columns="supplier", values=what)
    n = d.pivot(index="day", columns="supplier", values="lines")
    usual = x.shift(1).rolling(window, min_periods=window).median()
    moved = ((x - usual).abs() >= shift) & (n >= least)
    out = []
    for series in x.columns:
        quiet_until = None
        for day in moved.index[moved[series].to_numpy()]:
            if quiet_until is not None and day <= quiet_until:
                continue
            quiet_until = day + pd.Timedelta(days=window)
            out.append({"day": day, "supplier": series, "what": what,
                        "usual": usual.loc[day, series],
                        "today": x.loc[day, series],
                        "lines": int(n.loc[day, series])})
    cols = ["day", "supplier", "what", "usual", "today", "lines"]
    return pd.DataFrame(out, columns=cols)


def alerts(d: pd.DataFrame, first=None, last=None) -> pd.DataFrame:
    """Price and cost alerts between two days, oldest first."""
    a = pd.concat([check(d, "price"), check(d, "cost")])
    if first is not None:
        a = a[a.day >= pd.Timestamp(first)]
    if last is not None:
        a = a[a.day <= pd.Timestamp(last)]
    return a.sort_values(["day", "supplier"]).reset_index(drop=True)
