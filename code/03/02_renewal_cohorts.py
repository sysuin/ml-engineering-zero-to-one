# How many renewals reach their 90-day mark, when, and how many fail.
import json
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
# Contract end dates. Later contracts are kept unseen (see the brief).
FIRST, LAST = "2023-01-01", "2024-06-30"

renewals = pd.read_sql_query("""
    SELECT contract_id, account_id, end_date,
           date(end_date, '-90 days')  AS moment,
           outcome = 'not_renewed'     AS left_
    FROM contracts
    WHERE end_date BETWEEN ? AND ? AND outcome IS NOT NULL""",
    con, params=(FIRST, LAST))

n, leavers = len(renewals), int(renewals.left_.sum())
print(f"Renewals ending {FIRST} to {LAST}: {n:,}")
print(f"  not renewed          {leavers:>6,}  ({leavers / n:.1%})")
print(f"  'nobody leaves' is right about {1-leavers/n:.1%} of the time")

cohorts = renewals.groupby("moment").agg(
    contracts=("contract_id", "size"), leavers=("left_", "sum"))
months = len(cohorts)
weeks = months * 52 / 12
print(f"\nThe 90-day marks fall on {months} dates, one a month")
print(f"  contracts per cohort   mean {cohorts.contracts.mean():.0f}"
      f"   min {cohorts.contracts.min()}"
      f"   max {cohorts.contracts.max()}")
print(f"  leavers per cohort     mean {cohorts.leavers.mean():.1f}"
      f"   min {cohorts.leavers.min()}   max {cohorts.leavers.max()}")
print(f"  averaged over the weeks     {n / weeks:.0f} contracts,"
      f" {leavers / weeks:.1f} leavers a week")

# The same months, counted in the other units a table could use.
months_ = pd.period_range(cohorts.index.min(), cohorts.index.max(),
                          freq="M")
firsts = [str(m.start_time.date()) for m in months_]
account_months = sum(con.execute("""
    SELECT COUNT(DISTINCT account_id) FROM contracts
    WHERE start_date <= ? AND end_date >= ?""",
    (day, day)).fetchone()[0] for day in firsts)
orders = con.execute("""
    SELECT COUNT(*) FROM orders WHERE order_date BETWEEN ? AND ?""",
    (firsts[0], str(months_[-1].end_time.date()))).fetchone()[0]
print(f"\nRows a training table would have,"
      f" {months_[0]} to {months_[-1]}")
print(f"  one per contract renewal     {n:>9,}")
print(f"  one per account per month    {account_months:>9,}")
print(f"  one per order                {orders:>9,}")

# Contracts reaching the 90-day mark, week by week.
weekly = (pd.to_datetime(renewals.moment).dt.to_period("W-SUN")
          .value_counts().sort_index())
every_week = pd.period_range(weekly.index.min(), weekly.index.max(),
                             freq="W-SUN")
weekly = weekly.reindex(every_week, fill_value=0)
Path("code/03/02_renewal_cohorts.json").write_text(json.dumps({
    "weeks": [str(p.start_time.date()) for p in weekly.index],
    "contracts": [int(v) for v in weekly.values]}, indent=1))
