"""
Chapter 4's columns for a few contracts, quickly enough to answer a
request. Chapter 22 writes it.

at_mark() reads every order in the warehouse to make one cohort's
rows, which is right for a monthly job and far too slow for one
contract while somebody waits. rows() makes the same columns with the
same code, features.assemble(), but lets it see only the accounts it
was asked about. It opens the warehouse read-only, puts those accounts
in a temporary table, and shadows `orders` and `tickets` with
temporary views of their records alone. SQLite looks for a name among
temporary objects first, so every query in build_table.py reads the
views, and not one line of it changes.

The accounts include any legacy CRM id that belongs to them. Leaving
those out is Chapter 21's skew again, and listing 22/10 shows it.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import MOMENT_DAYS, legacy_ids
from foresight.pipeline.features import assemble

SCOPE = """
CREATE TEMP TABLE scope (account_id INTEGER PRIMARY KEY);
CREATE TEMP VIEW orders AS SELECT * FROM main.orders
    WHERE account_id IN (SELECT account_id FROM scope);
CREATE TEMP VIEW tickets AS SELECT * FROM main.tickets
    WHERE account_id IN (SELECT account_id FROM scope);
"""


def open_warehouse(warehouse: Path = ML_WAREHOUSE):
    """A connection that can read the warehouse and write nothing."""
    uri = f"{Path(warehouse).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def contracts(con, ids) -> pd.DataFrame:
    """The contracts asked for, each with its mark, as at_marks()
    reads them; with the account's name and whether it is a key
    account, and the outcome if one is on record."""
    ids = [int(i) for i in ids]
    rows = pd.read_sql_query(f"""
        SELECT c.contract_id, c.account_id, c.end_date,
               c.term_months, c.legacy_terms, c.discount_pct,
               c.outcome, a.name, a.is_key_account
        FROM contracts c JOIN accounts a USING (account_id)
        WHERE c.contract_id IN ({", ".join("?" * len(ids))})""",
        con, params=ids)
    rows["end_date"] = pd.to_datetime(rows.end_date)
    rows["moment"] = rows.end_date - pd.Timedelta(days=MOMENT_DAYS)
    rows["discount_pct"] = rows.discount_pct.astype("Int64")
    return rows


def scope(con, accounts) -> None:
    """Let the next queries see only these accounts' orders and
    tickets, under their current ids and any legacy ones."""
    wanted = {int(a) for a in accounts}
    wanted |= {old for old, new in legacy_ids(con).items()
               if new in wanted}
    con.executescript(SCOPE)
    con.executemany("INSERT INTO scope VALUES (?)",
                    [(a,) for a in sorted(wanted)])


def rows(ids, warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Chapter 4's columns for these contracts at their marks, by
    assemble(). Contracts not in the warehouse are left out."""
    with closing(open_warehouse(warehouse)) as con:
        found = contracts(con, ids)
        if found.empty:
            return found
        scope(con, found.account_id)
        keep = ["contract_id", "account_id", "end_date", "moment",
                "term_months", "legacy_terms", "discount_pct"]
        return assemble(con, found[keep])
