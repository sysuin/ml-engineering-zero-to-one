"""
Monthly units sold, arranged as panels: one row per month, one column
per series. Chapter 17 forecasts them.

A series is a slice of v_sales summed by calendar month: a category in
a region (25 series), or a product in a region (240). Two facts about
the warehouse shape every panel:

    The key accounts' orders are on record from January 2023 and the
    long tail's from January 2022. A 2022 total is the long tail alone,
    so a panel of all accounts starts in 2023. The long tail's own
    panel (strand="tail") starts in 2022, a year earlier.

    A product exists from its launch. Before it a series is missing
    (NaN), not zero: nobody failed to buy a product that was not sold.

    units = monthly_units(["category", "region"])
    units.loc["2024-07", ("Cleaning", "Midwest")]
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE

FIRST = {"all": "2023-01", "tail": "2022-01"}   # first month on record
LAST = "2025-12"
CATEGORY_REGION = ["category", "region"]
PRODUCT_REGION = ["category", "sku", "region"]


@lru_cache(maxsize=2)
def sales(warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Units by month, product, region and kind of account."""
    return _read(warehouse, "")


def account_sales(accounts: tuple,
                  warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """The same, for the named accounts only."""
    marks = ", ".join("?" * len(accounts))
    return _read(warehouse, f"WHERE v.account_id IN ({marks})",
                 accounts)


def _read(warehouse, where, params=()) -> pd.DataFrame:
    with sqlite3.connect(warehouse) as con:
        d = pd.read_sql_query(f"""
            SELECT substr(v.order_date, 1, 7) AS month, v.sku,
                   v.category, v.region, a.is_key_account AS key,
                   SUM(v.qty) AS units
            FROM v_sales v JOIN accounts a USING (account_id) {where}
            GROUP BY 1, 2, 3, 4, 5""", con, params=params)
        launched = pd.read_sql_query(
            "SELECT sku, launched_on FROM products", con)
    d["month"] = pd.PeriodIndex(d.month, freq="M")
    return d.merge(launched, on="sku")


def monthly_units(by: list[str], strand: str = "all",
                  drop_accounts: tuple = (),
                  warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """A panel of units: months down, one column per series in `by`.
    drop_accounts rebuilds the history as if they had never bought."""
    d = sales(warehouse)
    if drop_accounts:
        gone = account_sales(tuple(drop_accounts), warehouse)
        d = pd.concat([d, gone.assign(units=-gone.units)])
    if strand == "tail":
        d = d[d.key == 0]
    months = pd.period_range(FIRST[strand], LAST, freq="M")
    d = d[d.month >= months[0]]
    panel = (d.groupby(["month", *by]).units.sum()
             .unstack(by).reindex(months).fillna(0.0))
    # A series made only of launched products starts at the launch.
    launch = d.groupby(by).launched_on.agg(
        lambda s: None if s.isna().any() else s.min())
    opens = launch.dropna().map(lambda s: pd.Period(s[:7], freq="M"))
    for column, first in opens.items():
        panel.loc[panel.index < first, column] = np.nan
    panel.index.name = "month"
    return panel.sort_index(axis=1)


def established(panel: pd.DataFrame) -> pd.DataFrame:
    """The series with a value in the panel's first month."""
    return panel.loc[:, panel.iloc[0].notna()]
