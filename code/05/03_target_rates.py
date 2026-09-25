# The non-renewal rate overall, by group and cohort by cohort.
import json
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")


def rates(groups):
    """Rows, leavers, rate, and a rough 95% margin for each group."""
    out = groups.not_renewed.agg(rows="size", left="sum", rate="mean")
    out["margin"] = 1.96 * np.sqrt(out.rate * (1 - out.rate) / out.rows)
    return out


def show(title, table):
    print(title)
    for name, r in table.iterrows():
        print(f"  {name:16}{r.rows:>6,.0f}{r.left:>5,.0f}"
              f"{r.rate:>8.1%} ± {r.margin:.1%}")


overall = train.not_renewed
print(f"Training split: {overall.sum()} of {len(train):,} not"
      f" renewed, {overall.mean():.1%}")
show("By segment (rows, leavers, rate)",
     rates(train.groupby(train.segment.fillna("(missing)"))))
show("By region", rates(train.groupby("region")))

cohorts = rates(train.groupby("moment"))
print(f"By cohort: {len(cohorts)} cohorts, rates from"
      f" {cohorts.rate.min():.1%} to {cohorts.rate.max():.1%}")

# Small business either side of the July 2023 price change.
small = train[train.segment == "Small business"]
when = np.where(small.moment < "2023-07-01", "before Jul 2023",
                "from Jul 2023")
show("Small business, by the date of the mark",
     rates(small.groupby(when)))

# One row in the Midwest's spring 2024 renewals.
con = sqlite3.connect(ML_WAREHOUSE)
names = pd.read_sql_query("SELECT account_id, name FROM accounts", con)
con.close()
train = train.merge(names, on="account_id")
quarter = train.end_date.dt.to_period("Q").astype(str)
h = train[train.name.str.startswith("Halloway")].iloc[-1]
cell = train[(quarter == str(h.end_date.to_period("Q")))
             & (train.region == h.region)]
print(f"{h['name']}, {h.region}, ending {h.end_date:%Y-%m-%d}:")
print(f"  1 of {len(cell)} rows in its region and quarter,"
      f" {h.spend_365 / cell.spend_365.sum():.0%} of their spend")

by_rq = train.groupby([train.region, quarter]).not_renewed.agg(
    ["size", "sum"])
with open("code/05/03_target_rates.json", "w") as f:
    json.dump({"overall": round(overall.mean(), 4),
               "quarters": sorted(quarter.unique()),
               "cells": {f"{r}|{q}": [int(n), int(k)] for (r, q), (n, k)
                         in by_rq.iterrows()},
               "halloway": [h.region, str(h.end_date.to_period("Q")),
                            int(len(cell))]}, f)
