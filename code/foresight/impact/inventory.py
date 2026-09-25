"""
What a forecast is worth to the buyer who orders against it, in
dollars. Chapter 25 writes it.

A forecast changes what the warehouse holds, not what customers buy,
so its value can be measured by replaying the past: order what each
forecast said, month by month, and price what went wrong. A month that
sells less than was ordered leaves units on the shelf, which cost
money to hold until they sell. A month that sells more runs short, and
some of the shortfall is sales lost to a competitor.

Every price in StockCosts is an assumption, like the brief's save
rate, and is labelled as one wherever it is printed.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE


@dataclass(frozen=True)
class StockCosts:
    holding: float = 0.25     # a year's holding cost, per $ of cost
    months_held: float = 1.0  # how long a surplus unit waits to sell
    lost: float = 0.5         # share of a shortfall that is a lost sale

    def over(self, units, unit_cost) -> np.ndarray:
        """Dollars a surplus costs: holding it until it sells."""
        return (np.asarray(units) * np.asarray(unit_cost)
                * self.holding * self.months_held / 12)

    def under(self, units, margin) -> np.ndarray:
        """Dollars a shortfall costs: the margin on the lost sales."""
        return np.asarray(units) * np.asarray(margin) * self.lost


def prices(warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Each product's unit cost and its margin at list price."""
    with sqlite3.connect(warehouse) as con:
        p = pd.read_sql_query(
            "SELECT sku, unit_cost, list_price FROM products", con)
    return p.assign(margin=p.list_price - p.unit_cost).set_index("sku")


def cost(record: pd.DataFrame, order: str, costs: StockCosts,
         priced: pd.DataFrame) -> pd.Series:
    """Dollars of surplus and shortfall from ordering column `order`
    of a backtest `record` (series = (category, sku, region))."""
    sku = record.series.map(lambda s: s[1])
    unit_cost = priced.unit_cost.reindex(sku).to_numpy()
    margin = priced.margin.reindex(sku).to_numpy()
    gap = record[order].to_numpy() - record.actual.to_numpy()
    over = costs.over(np.clip(gap, 0, None), unit_cost).sum()
    under = costs.under(np.clip(-gap, 0, None), margin).sum()
    return pd.Series({"surplus": over, "shortfall": under,
                      "total": over + under})
