# The skew test: rows made for serving compared with the training
# table's, column by column, over every mark in the history. First the
# nightly SQL, then the shared definition, then tonight's cohort alone.
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, renewals
from foresight.pipeline.features import (SkewError, at_mark, at_marks,
                                         check_skew, mark_on, skew)

LIVE = Path(__file__).with_name("live.sql").read_text()
SQL_COLUMNS = ["days_since_order", "orders_90d", "orders_prev_90d",
               "spend_365", "tickets_90d"]
table = pd.read_parquet(TABLE)
marks = sorted(table.moment.unique())


def nightly(mark_list):
    """The SQL's columns for every contract reaching each mark."""
    with sqlite3.connect(ML_WAREHOUSE) as con:
        out = [pd.read_sql_query(LIVE, con, params={"mark": f"{m:%F}"})
               for m in map(pd.Timestamp, mark_list)]
    return pd.concat(out, ignore_index=True)


print(f"{len(marks)} marks, {len(table):,} contracts\n")
print("The nightly SQL against the training table")
try:
    check_skew(table, nightly(marks), SQL_COLUMNS)
except SkewError:
    found = skew(table, nightly(marks), SQL_COLUMNS)
    print(f"  {'column':<18}{'rows':>7}{'differ':>8}{'e.g.':>7}")
    for c, r in found.iterrows():
        e = "" if r.example is None else f"{r.example:.0f}"
        print(f"  {c:<18}{r.rows:>7,}{r.differ:>8,}{e:>7}")
    print("  SkewError: the test fails")

print("\nThe shared definition against the training table")
shared = at_marks(marks)
checked = check_skew(table, shared)
print(f"  {len(checked)} columns, {len(shared):,} contracts:"
      f" {checked.differ.sum()} differences")

tonight = mark_on("2026-01-01")
cohort = at_mark(tonight)
print(f"\nTonight's list, marked {tonight:%Y-%m-%d}:"
      f" {len(cohort)} contracts")
with sqlite3.connect(ML_WAREHOUSE) as con:
    end = f"{tonight + pd.Timedelta(days=90):%F}"
    labelled = renewals(con, end, end)
print(f"  the training table's query finds {len(labelled)} of them")
alone = skew(cohort, nightly([tonight]), SQL_COLUMNS)
print("  the SQL against the shared definition, tonight only:")
for c, r in alone.iterrows():
    print(f"  {c:<18}{int(r.rows):>7,}{int(r.differ):>8,}")
