"""
What an account spends between its 90-day mark and the end of its
contract: the number Chapter 6 learns to predict.

The rows and features are the Chapter 4 table's, read from its Parquet
file. Two columns are added from the orders, filed under each account's
current id exactly as the builder files them:

    spend_90d        spent in the 90 days before the mark   (a feature)
    spend_next_90d   spent from the mark until the contract ends
                     (the target, known only once the contract ends)

The splits are DECISIONS.md's, by end date. The test year is not read.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, events, legacy_ids

TRAIN = ("2023-01-01", "2024-06-30")
VALIDATION = ("2024-07-01", "2024-12-31")
FEATURES = ["spend_90d", "spend_365", "orders_90d", "orders_prev_90d",
            "days_since_order", "tickets_90d", "tenure_days",
            "term_months", "legacy_terms"]


def spend_table(first: str, last: str, table: Path = TABLE,
                warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Chapter 4's rows ending in [first, last], with spend around."""
    rows = pd.read_parquet(table)
    rows = rows[rows.end_date.between(first, last)]
    con = sqlite3.connect(warehouse)
    orders, _ = events(con, legacy_ids(con))
    key = pd.read_sql_query(
        "SELECT account_id, is_key_account FROM accounts", con)
    con.close()

    o = rows[["contract_id", "account_id", "moment", "end_date"]].merge(
        orders, on="account_id")
    o["age"] = (o.moment - o.day).dt.days       # 1 = the day before
    before = o[(o.age >= 1) & (o.age <= 90)]
    after = o[(o.day >= o.moment) & (o.day < o.end_date)]

    rows = rows.merge(key, on="account_id", how="left")
    for name, part in (("spend_90d", before),
                       ("spend_next_90d", after)):
        total = part.groupby("contract_id").value.sum()
        rows[name] = rows.contract_id.map(total).fillna(0.0).round(2)
    return rows.reset_index(drop=True)


def features(rows: pd.DataFrame) -> np.ndarray:
    """The numeric columns as a grid, one row per contract."""
    X = rows[FEATURES].astype(float)
    # No order before the mark (a handful of rows): call it a year,
    # as far back as any window in the table looks.
    X["days_since_order"] = X.days_since_order.fillna(365)
    return X.to_numpy()


def year_on_record(rows: pd.DataFrame,
                   warehouse: Path = ML_WAREHOUSE) -> pd.Series:
    """True where the year before the mark is all inside the record."""
    with sqlite3.connect(warehouse) as con:
        starts = pd.read_sql_query("""
            SELECT a.is_key_account, MIN(o.order_date) AS first_day
            FROM orders o JOIN accounts a USING (account_id)
            GROUP BY a.is_key_account""", con)
    first_day = pd.to_datetime(
        starts.set_index("is_key_account").first_day)
    window_opens = rows.moment - pd.Timedelta(days=365)
    return window_opens >= rows.is_key_account.map(first_day)


# ------------------------------------------------ baselines and scores
def baselines(train: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Three predictions for rows, each needing no model at all."""
    return {
        "mean of the training rows":
            np.full(len(rows), train.spend_next_90d.mean()),
        "last value: the 90 days before":
            rows.spend_90d.to_numpy(),
        "run rate: last year x 90/365":
            rows.spend_365.to_numpy() * 90 / 365,
    }


def mae(y, pred) -> float:
    """Mean absolute error: the average miss, in dollars."""
    return float(np.mean(np.abs(np.asarray(y) - pred)))


def rmse(y, pred) -> float:
    """Root mean squared error: big misses count for more."""
    return float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))


def scores(y, predictions: dict) -> None:
    """Print MAE and RMSE for each prediction, one line each."""
    print(f"  {'':<34}{'MAE':>10}{'RMSE':>10}")
    for name, pred in predictions.items():
        print(f"  {name:<34}{mae(y, pred):>10,.0f}"
              f"{rmse(y, pred):>10,.0f}")
