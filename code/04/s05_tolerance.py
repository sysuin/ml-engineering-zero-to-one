# Exercise 5: merge_asof with a tolerance of a year.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
renewals = pd.read_sql_query("""
    SELECT contract_id, account_id,
           date(end_date, '-90 days') AS moment,
           outcome = 'not_renewed' AS left_
    FROM contracts
    WHERE end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND outcome IS NOT NULL""", con, parse_dates=["moment"])
orders = pd.read_sql_query("SELECT account_id, order_date FROM orders",
                           con, parse_dates=["order_date"])

for tolerance in (None, pd.Timedelta(days=365)):
    t = pd.merge_asof(renewals.sort_values("moment"),
                      orders.sort_values("order_date"),
                      left_on="moment", right_on="order_date",
                      by="account_id", allow_exact_matches=False,
                      tolerance=tolerance)
    t["days"] = (t.moment - t.order_date).dt.days
    top = (t.sort_values(["moment", "days", "contract_id"],
                         ascending=[True, False, True],
                         na_position="last")
            .groupby("moment").head(40))
    name = "no tolerance" if tolerance is None else "a year"
    print(f"{name:13} no match {t.order_date.isna().sum():>4}"
          f"   rule: {top.left_.sum()} leavers, {top.left_.mean():.1%}")
