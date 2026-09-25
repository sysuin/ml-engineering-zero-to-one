"""
What the features read, and the one rule for reading it. Chapter 12.

load_sources() reads the warehouse once: orders with their value,
order lines summed by category and by supplier, tickets with their
category and body, and the accounts' fixed facts. Every record filed
under a legacy CRM id is moved to the account's current id with
Chapter 4's own mapping, and a record the mapping cannot reach stops
the build, as it does in build_table.

A Context pairs the sources with the contracts to describe, and hands
features their events through window(), which keeps only records dated
strictly before each contract's mark: age 1 is the day before, and the
mark itself is age 0, too late. Features never filter dates themselves.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import (LEGACY_FROM, TableCheckError,
                                        events, legacy_ids)

LONGEST = 365           # the longest window any feature may read

# The first day each population's orders are on record (Chapter 4).
RECORD_STARTS = {0: pd.Timestamp("2022-01-01"),     # the long tail
                 1: pd.Timestamp("2023-01-01")}     # key accounts


@dataclass
class Sources:
    orders: pd.DataFrame        # account_id, day, value
    lines: pd.DataFrame         # account_id, day, category, supplier,
                                # value: one row per order line group
    tickets: pd.DataFrame       # account_id, day, category, sku, body
    accounts: pd.DataFrame      # account_id, since, is_key_account,
                                # postcode


def _current_ids(df: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    df["account_id"] = df.account_id.replace(id_map)
    stranded = int((df.account_id >= LEGACY_FROM).sum())
    if stranded:
        raise TableCheckError(f"{stranded:,} events on legacy ids")
    return df


def load_sources(warehouse: Path = ML_WAREHOUSE) -> Sources:
    """Everything the library reads, under current account ids."""
    con = sqlite3.connect(warehouse)
    id_map = legacy_ids(con)
    orders, _ = events(con, id_map)
    lines = pd.read_sql_query("""
        SELECT o.account_id, o.order_date AS day, p.category,
               s.name AS supplier, SUM(l.qty * l.unit_price) AS value
        FROM order_lines l
        JOIN orders o USING (order_id)
        JOIN products p USING (sku)
        JOIN suppliers s USING (supplier_id)
        GROUP BY o.account_id, o.order_date, p.category, s.name""",
        con)
    tickets = pd.read_sql_query("""
        SELECT account_id, date(opened_at) AS day, category, sku, body
        FROM tickets""", con)
    accounts = pd.read_sql_query("""
        SELECT account_id, since, is_key_account, postcode
        FROM accounts WHERE crm_source = 'Meridian CRM'""", con)
    con.close()
    for df in (lines, tickets):
        _current_ids(df, id_map)
        df["day"] = pd.to_datetime(df.day)
    accounts["since"] = pd.to_datetime(accounts.since)
    return Sources(orders, lines, tickets, accounts)


class Context:
    """The contracts to describe, and their sources, with one way of
    reaching an event: window()."""

    def __init__(self, keys: pd.DataFrame, sources: Sources):
        self.keys = keys.reset_index(drop=True)
        self.sources = sources
        self.index = pd.Index(self.keys.contract_id, name="contract_id")
        self._memo: dict = {}

    def window(self, source: str, days: int) -> pd.DataFrame:
        """Each contract's events from `source` dated 1 to `days` days
        before its mark, with their age in days."""
        if not 1 <= days <= LONGEST:
            raise ValueError(f"windows run from 1 to {LONGEST} days")
        if source not in self._memo:
            events = getattr(self.sources, source)
            k = self.keys[["contract_id", "account_id", "moment"]]
            e = k.merge(events, on="account_id")
            e["age"] = (e.moment - e.day).dt.days
            self._memo[source] = e[(e.age >= 1) & (e.age <= LONGEST)]
        e = self._memo[source]
        return e[e.age <= days]

    def per_contract(self, values: pd.Series, fill=0.0) -> pd.Series:
        """Values keyed by contract_id, one per contract, in order."""
        return values.reindex(self.index, fill_value=fill).astype(float)

    def column(self, name: str) -> pd.Series:
        """A column of the Chapter 4 table, keyed by contract_id."""
        return self.keys.set_index("contract_id")[name]
