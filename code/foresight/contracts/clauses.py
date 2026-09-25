"""
What a data contract says, and the check that reads it. Chapter 23.

A contract is an agreement between the team that writes a table and
the teams that read it: which columns, of which types, holding which
values, how fresh, how many rows a day, and the promises that only a
query across rows can check. breaches() checks every clause against a
database as it stood on the morning of `on` and returns one Breach per
clause broken; an empty list means the table kept its side.

Promises no query can see, such as "a corrected row is a new row, never
an edit", are kept in the contract's words. They are for the
conversation with the owner, and the check does not pretend to test
them.
"""
from __future__ import annotations

import sqlite3
import textwrap
from dataclasses import dataclass, field

import pandas as pd

WINDOW = 28             # days over which the daily volume is checked


@dataclass(frozen=True)
class Column:
    name: str
    type: str                   # SQLite's typeof(): integer, real, text
    required: bool = True       # never NULL
    values: tuple = ()          # the only values allowed
    low: float | None = None    # the smallest allowed
    high: float | None = None   # the largest allowed
    form: str | None = None     # a GLOB pattern every value matches
    refers: str | None = None   # "table.column" every value exists in
    why: str = ""               # what in Foresight depends on it


@dataclass(frozen=True)
class Rule:
    promise: str                # in words
    sql: str                    # counts the rows that break it


@dataclass(frozen=True)
class Contract:
    table: str
    owner: str                  # the team that writes the table
    version: str                # changed only by agreement
    columns: tuple
    key: tuple                  # no two rows share these
    when: str | None = None     # the column that dates each row
    fresh_days: int | None = None   # latest row at most this old
    daily_rows: int | None = None   # fewest rows on any recent day
    rules: tuple = ()
    words: tuple = ()
    readers: tuple = field(default=())  # what in Foresight reads it


@dataclass(frozen=True)
class Breach:
    table: str
    clause: str
    rows: int
    example: str = ""
    unit: str = "rows"

    def evidence(self) -> str:
        """How many rows (or days) broke it, and one of them."""
        n = f"{self.rows:,} {self.unit}" if self.unit else ""
        eg = f"e.g. {self.example}" if self.example else ""
        return "; ".join(x for x in (n, eg) if x)

    def line(self) -> str:
        return f"{self.table}: {self.clause} ({self.evidence()})"


def _rows(c: Contract, on) -> str:
    """The table as it stood on the morning of `on`."""
    if c.when is None or on is None:
        return f"SELECT * FROM {c.table}"
    day = f"'{pd.Timestamp(on):%Y-%m-%d}'"
    # A date that will not parse stays in, so its column's form fails.
    return (f"SELECT * FROM {c.table} WHERE date({c.when}) < {day}"
            f" OR date({c.when}) IS NULL")


def _count(con, rows: str, where: str, show: str) -> tuple[int, str]:
    """How many rows match `where`, and one of them."""
    n, eg = con.execute(
        f"SELECT COUNT(*), MIN({show}) FROM ({rows}) WHERE {where}"
    ).fetchone()
    return n, "" if eg is None else str(eg)


def _column_clauses(col: Column):
    """(clause, SQL condition for a row that breaks it)."""
    n, out = col.name, []
    kinds = [col.type] + ([] if col.required else ["null"])
    out.append((f"{n} is {col.type}", "typeof(" + n + ") NOT IN ("
                + ", ".join(f"'{k}'" for k in kinds) + ")"))
    if col.required:
        out.append((f"{n} is never empty", f"{n} IS NULL"))
    if col.values:
        listed = ", ".join(repr(v) for v in col.values)
        out.append((f"{n} is one of {len(col.values)}",
                    f"{n} NOT IN ({listed})"))
    if col.low is not None:
        out.append((f"{n} >= {col.low:g}", f"{n} < {col.low}"))
    if col.high is not None:
        out.append((f"{n} <= {col.high:g}", f"{n} > {col.high}"))
    if col.form:
        out.append((f"{n} looks like {shape(col.form)}",
                    f"NOT {n} GLOB '{col.form}'"))
    if col.refers:
        table, other = col.refers.split(".")
        out.append((f"{n} is in {col.refers}",
                    f"{n} NOT IN (SELECT {other} FROM {table})"))
    return out


def breaches(con: sqlite3.Connection, c: Contract, on=None
             ) -> list[Breach]:
    """Every clause of `c` the table breaks on the morning of `on`."""
    rows, found = _rows(c, on), []
    show = c.key[0]
    have = {r[1] for r in con.execute(f"PRAGMA table_info({c.table})")}
    missing = [col.name for col in c.columns if col.name not in have]
    for name in missing:
        found.append(Breach(c.table, f"has a column {name}", 1))
    for col in c.columns:
        if col.name in missing:
            continue
        for clause, where in _column_clauses(col):
            n, eg = _count(con, rows, where, show)
            if n:
                found.append(Breach(c.table, clause, n, eg))
    keys = ", ".join(c.key)
    n, = con.execute(f"SELECT COUNT(*) FROM (SELECT 1 FROM ({rows})"
                     f" GROUP BY {keys} HAVING COUNT(*) > 1)"
                     ).fetchone()
    if n:
        found.append(Breach(c.table, f"one row per {keys}", n))
    for rule in c.rules:
        n, = con.execute(rule.sql.format(rows=f"({rows})")).fetchone()
        if n:
            found.append(Breach(c.table, rule.promise, n))
    if on is not None and c.when is not None:
        found += _timeliness(con, c, rows, pd.Timestamp(on))
    return found


def _timeliness(con, c: Contract, rows: str, on) -> list[Breach]:
    """Freshness: the latest row is recent enough. Volume: every one
    of the last WINDOW days has at least daily_rows rows."""
    out = []
    (last,) = con.execute(f"SELECT MAX(date({c.when}))"
                          f" FROM ({rows})").fetchone()
    if c.fresh_days is not None and (
            last is None or pd.Timestamp(last)
            < on - pd.Timedelta(days=c.fresh_days)):
        out.append(Breach(c.table, f"rows to within {c.fresh_days}"
                          " day(s) of the morning", 1,
                          f"latest {last}", unit=""))
    if c.daily_rows is not None:
        first = on - pd.Timedelta(days=WINDOW)
        days = pd.date_range(first, on - pd.Timedelta(days=1))
        got = dict(con.execute(
            f"SELECT date({c.when}), COUNT(*) FROM ({rows})"
            f" WHERE date({c.when}) >= '{first:%Y-%m-%d}'"
            " GROUP BY 1").fetchall())
        short = [f"{d:%Y-%m-%d}: {got.get(f'{d:%Y-%m-%d}', 0)}"
                 for d in days
                 if got.get(f"{d:%Y-%m-%d}", 0) < c.daily_rows]
        if short:
            out.append(Breach(c.table, f"at least {c.daily_rows:,}"
                              f" rows a day for {WINDOW} days",
                              len(short), short[0], unit="day(s)"))
    return out


def clauses(c: Contract) -> int:
    """How many clauses the check reads, for a report."""
    n = sum(len(_column_clauses(col)) for col in c.columns)
    n += 1 + len(c.rules)
    n += (c.fresh_days is not None) + (c.daily_rows is not None)
    return n


def describe(c: Contract) -> str:
    """The contract as a page a person can read, 68 wide."""
    out = [f"{c.table}  v{c.version}  owner: {c.owner}"]
    for col in c.columns:
        rest = []
        if col.values:
            rest += textwrap.wrap("one of " + ", ".join(
                map(str, col.values)), 68 - 26)
        if col.low is not None and col.high is not None:
            rest.append(f"{col.low:g} to {col.high:g}")
        elif col.low is not None:
            rest.append(f">= {col.low:g}")
        if col.form:
            rest.append("like " + shape(col.form))
        if col.refers:
            rest.append("in " + col.refers)
        kind = col.type + ("" if col.required else "?")
        first = rest.pop(0) if rest else ""
        out.append(f"  {col.name + ' ':<14}{kind:<10}{first}".rstrip())
        out += [f"  {'':<24}{r}" for r in rest]
    out.append(f"  one row per {', '.join(c.key)}")
    if c.fresh_days is not None:
        out.append(f"  fresh: rows to {c.fresh_days} day(s) before"
                   " the morning")
    if c.daily_rows is not None:
        out.append(f"  volume: {c.daily_rows:,}+ rows on each of"
                   f" the last {WINDOW} days")
    for head, lines in (("rule", [r.promise for r in c.rules]),
                        ("promise", c.words)):
        for text in lines:
            out += textwrap.wrap(f"{head}: {text}", 68,
                                 initial_indent="  ",
                                 subsequent_indent="    ")
    return "\n".join(out)


def shape(form: str) -> str:
    """A GLOB pattern as a person would write it: 9 for a digit, A for
    a capital letter."""
    return (form.replace("[0-9]", "9").replace("[A-Z]", "A")
                .replace("[[]", "[").replace("[]]", "]"))
