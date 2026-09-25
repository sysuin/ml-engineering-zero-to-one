"""
Data contracts: what Foresight relies on in each table it reads, as
agreed with the team that writes it, and what it promises in the one
table others read from it. Chapter 23 writes them.

    python -m foresight.contracts                 the feeds, today
    python -m foresight.contracts --on 2025-12-31

    clauses.py    Column, Rule, Contract; breaches(), describe()
    orders.py     orders and order_lines: the order platform team
    tickets.py    tickets: the support desks
    answers.py    the scores database's answers view: Foresight

check() opens the warehouse read-only, checks every feed's contract
as the tables stood on the morning of `on`, and raises ContractError
naming every clause broken. The day comes from --on, or FORESIGHT_ON,
as the monthly job's does.
"""
from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

from foresight.config import ML_WAREHOUSE
from foresight.contracts.answers import ANSWERS
from foresight.contracts.clauses import (Breach, Column, Contract, Rule,
                                         breaches, clauses, describe)
from foresight.contracts.orders import ORDER_LINES, ORDERS
from foresight.contracts.tickets import TICKETS

FEEDS = (ORDERS, ORDER_LINES, TICKETS)

__all__ = ["ANSWERS", "FEEDS", "ORDERS", "ORDER_LINES", "TICKETS",
           "Breach", "Column", "Contract", "ContractError", "Rule",
           "breaches", "check", "clauses", "describe", "read_only"]


class ContractError(Exception):
    """A feed broke its contract."""


def read_only(path: Path) -> sqlite3.Connection:
    """A connection that cannot write, as every check should hold."""
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro",
                           uri=True)


def check(on, warehouse: Path = ML_WAREHOUSE,
          contracts=FEEDS) -> list[Breach]:
    """Every clause broken by any feed; raise if there is one."""
    with closing(read_only(warehouse)) as con:
        found = [b for c in contracts for b in breaches(con, c, on)]
    if found:
        raise ContractError("\n".join(b.line() for b in found))
    return found


def main(argv=None) -> None:
    from foresight.serve import today
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--on", default=None, help="the morning, as"
                    " YYYY-MM-DD; FORESIGHT_ON or today if not given")
    ap.add_argument("--warehouse", type=Path, default=ML_WAREHOUSE)
    args = ap.parse_args(argv)
    on = args.on or today()
    check(on, args.warehouse)
    n = sum(clauses(c) for c in FEEDS)
    print(f"{len(FEEDS)} feeds, {n} clauses kept on the morning of"
          f" {str(on)[:10]}")
