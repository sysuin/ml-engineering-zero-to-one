# Write the exploration's findings as a short report, numbers included.
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE, ROOT
from foresight.data.build_table import TABLE
from foresight.data.expectations import EXPECTATIONS, failures

table = pd.read_parquet(TABLE)
train = table[table.end_date <= "2024-06-30"]
con = sqlite3.connect(ML_WAREHOUSE)
key = pd.read_sql_query(
    "SELECT account_id, is_key_account AS key FROM accounts", con)
con.close()
t = train.merge(key, on="account_id")
y = t.not_renewed
cohort = t.groupby("moment").not_renewed.mean()
q1, q3 = t.days_since_order.astype(float).quantile([0.25, 0.75])
far = t.days_since_order > q3 + 1.5 * (q3 - q1)
seg = t.groupby("segment").not_renewed.mean()
reg = t.groupby("region").not_renewed.mean()
gap = t.discount_pct.isna()
none = t.days_since_order.isna()
ka = t[t.key == 1]
by_mark = ka.groupby("moment").spend_365.median()
small = t[t.segment == "Small business"]
sb = small.groupby(small.moment >= "2023-07-01").not_renewed.mean()

findings = [
    (f"{y.sum()} of {len(t):,} renewals were not renewed"
     f" ({y.mean():.1%}); cohorts ran from {cohort.min():.1%} to"
     f" {cohort.max():.1%}",
     "label rate per split"),
    (f"Recency's box-plot outliers are the leavers: past"
     f" {q3 + 1.5 * (q3 - q1):.0f} days, {y[far].mean():.1%} left;"
     f" otherwise {y[~far].mean():.1%}", "recency range"),
    (f"Spend is skewed (skew {t.spend_365.skew():.0f}; on a log scale"
     f" {np.log10(t.spend_365[t.spend_365 > 0]).skew():.2f}); the"
     f" {len(ka)} key-account rows hold"
     f" {ka.spend_365.sum() / t.spend_365.sum():.0%} of it. Not capped",
     "no negatives"),
    (f"Small business leaves most ({seg.max():.1%}), enterprise least"
     f" ({seg.min():.1%}); regions differ less than their margins"
     f" ({reg.min():.1%} to {reg.max():.1%})", "allowed values"),
    (f"discount_pct is missing on {gap.sum():,} rows, exactly the"
     f" legacy contracts, which left at {y[gap].mean():.1%} against"
     f" {y[~gap].mean():.1%}. Never fill it", "missing iff legacy"),
    (f"{none.sum()} rows have no order before the mark,"
     f" {(none & (t.key == 1)).sum()} of them key accounts marked"
     f" before their record starts: their zeros mean no record",
     "zeros agree with dates"),
    (f"Key-account spend covers under a year until 2024: its median"
     f" rose from {by_mark[by_mark > 0].iloc[0]:,.0f} to"
     f" {by_mark.iloc[-1]:,.0f}",
     "none yet: needs a column"),
    (f"Small business left at {sb[False]:.1%} before July 2023 and"
     f" {sb[True]:.1%} after, inside the noise", "none: Chapter 8"),
]
DECISIONS = [
    "- Keep the discount gaps. `legacy_terms` is their indicator;"
    " filling with 0 or the mean erases what the gap says.",
    "- Give models spend on a log scale. Keep the key accounts: their"
    " size is real, and one of them is the most expensive loss.",
    "- Read the zeros of the no-order rows as missing, not as nothing.",
    "- Report leavers by row and by money, separately. Halloway is one"
    " row and most of its region's spend.",
]
OPEN = [
    "- Key accounts' windows are short before 2024. Add a column for"
    " how much history each row has (Chapter 12)?",
    "- Did the small-business rate move after July 2023? Too small to"
    " tell here (Chapter 8).",
    "- Why do legacy contracts carry no discount? Confirm with whoever"
    " owns the contracts table.",
]
lines = ["# Renewals table: what exploration found", "",
         f"Training split only: {len(t):,} of {len(table):,} rows,"
         " contracts ending 2023-01-01 to 2024-06-30. Validation and"
         " test labels were not looked at.", "",
         "## Question", "",
         "What in this table is related to a contract not renewing, and"
         " which values cannot be taken at face value?", "",
         "## Findings", "", "| | Finding | Checked by |",
         "|---|---|---|"]
lines += [f"| {i} | {f} | {c} |" for i, (f, c) in
          enumerate(findings, 1)]
lines += ["", "## Decisions", "", *DECISIONS, "",
          "## Open questions", "", *OPEN, "", "## Checks", "",
          f"{len(EXPECTATIONS)} expectations in"
          " `foresight/data/expectations.py`, run on every table the"
          " builder writes; broken on this one:"
          f" {len(failures(table))}. Rerun:"
          " `python -m foresight.data.expectations`.", ""]
path = ROOT / "docs" / "foresight-eda.md"
path.write_text("\n".join(lines))
words = len(path.read_text().split())
print(f"Wrote {Path('docs', path.name)}: {words} words")
for line in lines:
    if line.startswith("## "):
        print(f"  {line[3:]}")
checked = sum(not c.startswith("none") for _, c in findings)
print(f"{len(findings)} findings, {checked} with a check;"
      f" {len(failures(table))} checks broken")
