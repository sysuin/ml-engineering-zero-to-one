"""
Foresight's training job, v0.5. Chapter 13 writes it; Chapter 21 turns
it into a pipeline.

    python -m foresight.train                       as of 1 January 2025
    python -m foresight.train --as-of 2024-07-01

One run does five things, in order, and stops at the first that fails:

1. reads the training table and checks Chapter 5's expectations;
2. keeps the contracts whose outcome was on record on the day;
3. backtests the model on the latest six cohorts it can see;
4. runs the leakage checks (checks/leakage.py) on every column the
   model reads, against the cohort whose list is made that day;
5. fits v0.5's model on every outcome it may use.

No model is fitted if a check fails. v0.5's model is v0.4's lasso on
Chapter 4's columns: Chapter 12's rerun of the leaderboard left it in
place, and Chapter 13 found no reason to change it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from foresight.checks.leakage import run_checks
from foresight.data.build_table import TABLE
from foresight.data.expectations import check_table
from foresight.evaluate import HISTORY_FROM, backtest, known_by
from foresight.features.sources import load_sources
from foresight.models.featured import FeaturedLasso

STRENGTH = 0.002        # Chapter 9's choice, kept by Chapters 11 and 12
COLUMNS = ["days_since_order", "orders_90d", "orders_prev_90d",
           "spend_365", "tickets_90d", "tenure_days", "segment",
           "region", "term_months", "legacy_terms", "discount_pct"]
"""The table columns the model reads (Chapter 7 turns them into
sixteen numbers)."""
BACKTEST_COHORTS = 6


def make_model(extra=()):
    """v0.5's model, with any extra columns a caller asks for."""
    return lambda: FeaturedLasso(extra, STRENGTH)


def train(as_of: str = "2025-01-01", table: pd.DataFrame | None = None,
          extra: tuple = (), sources=None, allow: dict | None = None
          ) -> dict:
    """Check, then fit, on the outcomes known on `as_of`. `extra`
    names more columns of `table` for the model to read; they go
    through every check like the rest."""
    table = pd.read_parquet(TABLE) if table is None else table
    check_table(table)
    history = table[table.end_date >= HISTORY_FROM]
    rows = known_by(history, as_of)
    marks = np.sort(table.moment.unique())
    today = pd.Timestamp(marks[marks <= pd.Timestamp(as_of)][-1])
    recent = table[table.moment == today]      # the list being made

    ends = np.sort(rows.end_date.unique())[-BACKTEST_COHORTS:]
    first, last = (str(pd.Timestamp(d).date()) for d in ends[[0, -1]])
    scored = backtest(rows, first, last, make_model(extra))
    sources = load_sources() if sources is None else sources
    report = run_checks(rows, recent, COLUMNS + list(extra), sources,
                        scored=scored, allow=allow)
    model = make_model(extra)().fit(rows)
    return {"model": model, "report": report, "as_of": as_of,
            "rows": len(rows), "recent": len(recent), "mark": today}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default="2025-01-01")
    args = ap.parse_args(argv)
    run = train(args.as_of)
    print(run["report"].text())
    print(f"\nfitted on {run['rows']:,} outcomes known on"
          f" {args.as_of}; ready to score the {run['recent']} contracts"
          f" marked {run['mark']:%Y-%m-%d}")


if __name__ == "__main__":
    main()
