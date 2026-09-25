"""
What must be true of every renewals table, learned by looking at one.

    python -m foresight.data.expectations              the table on disk
    python -m foresight.data.expectations --table other.parquet

Chapter 4's check() guards the builder's structure: row counts, keys,
dates. These guard the content. Each expectation is a finding from
Chapter 5's exploration of the training split, written as a rule that
any table the builder produces must obey. check_table() runs them all
and raises ExpectationError, naming every rule broken and a few of the
rows that broke it. Run it after build_table.write(), on every table.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from foresight.data.build_table import TABLE

SEGMENTS = {"Small business", "Mid-market", "Public sector",
            "Enterprise"}
REGIONS = {"Northeast", "Southeast", "Midwest", "Southwest", "West"}
DISCOUNTS = {0, 3, 5, 8, 10, 12, 15}
TERMS = {12, 24}
ORDERS_FROM = pd.Timestamp("2022-01-01")    # the first order on record
KEY_HISTORY_FROM = pd.Timestamp("2023-01-01")   # key accounts' record
SPLITS = {"train": ("2023-01-01", "2024-06-30"),
          "validation": ("2024-07-01", "2024-12-31"),
          "test": ("2025-01-01", "2025-12-31")}
LABEL_RATE = (0.03, 0.13)   # half to twice the training split's 6.5%


class ExpectationError(Exception):
    """The table broke at least one expectation."""


@dataclass(frozen=True)
class Expectation:
    """A rule, and a function giving True wherever the rule holds."""
    rule: str
    holds: Callable[[pd.DataFrame], pd.Series]


def split_of(end_date: pd.Series) -> pd.Series:
    """The name of the split each row's end date falls in."""
    out = pd.Series(pd.NA, index=end_date.index, dtype="object")
    for name, (first, last) in SPLITS.items():
        out[end_date.between(first, last)] = name
    return out


def label_rate_by_split(t: pd.DataFrame) -> pd.Series:
    rates = t.not_renewed.groupby(split_of(t.end_date)).mean()
    return rates.between(*LABEL_RATE)


def days_known(t: pd.DataFrame) -> pd.Series:
    """Days between the first order on record and each row's moment."""
    return (t.moment - ORDERS_FROM).dt.days


def quiet_for(t: pd.DataFrame, days: int) -> pd.Series:
    """True where the account placed no order in the last `days`."""
    d = t.days_since_order
    return (d.isna() | (d > days)).fillna(True).astype(bool)


EXPECTATIONS = [
    # Ranges: values no account can have.
    Expectation("days_since_order runs from 1 to the first order on"
                " record",
                lambda t: t.days_since_order.isna()
                | t.days_since_order.between(1, days_known(t))),
    Expectation("counts and spend are never negative",
                lambda t: (t[["orders_90d", "orders_prev_90d",
                              "tickets_90d", "spend_365"]] >= 0)
                .all(axis=1)),
    Expectation("tenure_days is positive", lambda t: t.tenure_days > 0),
    # Allowed values.
    Expectation("segment is one of four, or missing",
                lambda t: t.segment.isna() | t.segment.isin(SEGMENTS)),
    Expectation("region is one of five",
                lambda t: t.region.isin(REGIONS)),
    Expectation("term_months is 12 or 24",
                lambda t: t.term_months.isin(TERMS)),
    Expectation("discount_pct is on the price list, or missing",
                lambda t: t.discount_pct.isna()
                | t.discount_pct.isin(DISCOUNTS)),
    # Missingness: every gap has a reason, and the reason is checked.
    Expectation("discount_pct is missing exactly when legacy_terms"
                " is 1",
                lambda t: t.discount_pct.isna()
                == (t.legacy_terms == 1)),
    Expectation("segment is missing only before key accounts' history",
                lambda t: t.segment.notna()
                | (t.moment < KEY_HISTORY_FROM)),
    # Zeros that must agree with the dates they summarise.
    Expectation("spend_365 is 0 exactly when no order in the year",
                lambda t: (t.spend_365 == 0) == quiet_for(t, 365)),
    Expectation("orders_90d is 0 exactly when no order in 90 days",
                lambda t: (t.orders_90d == 0) == quiet_for(t, 90)),
    # The label: a rate per split, far from what the training split had
    # only if the label or its join is broken.
    Expectation("the label rate in each split is between 3% and 13%",
                label_rate_by_split),
]


def failures(table: pd.DataFrame) -> list[str]:
    """One line per broken expectation. Empty means the table passed."""
    t = table.set_index("contract_id", drop=False)
    found = []
    for e in EXPECTATIONS:
        holds = e.holds(t).astype(bool)
        broken = holds.index[~holds.to_numpy()]
        if len(broken):
            shown = ", ".join(str(b) for b in broken[:3])
            more = ", ..." if len(broken) > 3 else ""
            found.append(f"{e.rule}\n    fails for {len(broken):,}:"
                         f" {shown}{more}")
    return found


def check_table(table: pd.DataFrame) -> None:
    """Raise ExpectationError if the table breaks any expectation."""
    found = failures(table)
    if found:
        n = len(EXPECTATIONS)
        head = f"{len(found)} of {n} expectations broken"
        raise ExpectationError("\n  ".join([head, *found]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", type=Path, default=TABLE)
    args = ap.parse_args(argv)
    table = pd.read_parquet(args.table)
    check_table(table)
    print(f"{len(EXPECTATIONS)} expectations hold on {len(table):,}"
          f" rows of {args.table.name}")


if __name__ == "__main__":
    main()
