"""
Making forecasts add up: product, category, region and total.

Forecasts here are wide frames: one row per forecast (a horizon and a
target month), one column per series. The bottom level is product x
region, with columns (category, sku, region); every level above it is
a sum of bottom columns.

    aggregate(bottom, ["region"])        bottom-up: add the products
    proportional(bottom, parent, by)     scale each group of products
                                         so it sums to its parent
    top_down(total, shares)              split one number by shares
"""
from __future__ import annotations

import pandas as pd

LEVELS = {"product x region": ["category", "sku", "region"],
          "category x region": ["category", "region"],
          "region": ["region"],
          "total": []}


def aggregate(bottom: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum the bottom columns into the level `by` ([] is the total)."""
    if not by:
        return bottom.sum(axis=1).to_frame("total")
    return bottom.T.groupby(level=by).sum().T


def proportional(bottom: pd.DataFrame, parent: pd.DataFrame,
                 by: list[str]) -> pd.DataFrame:
    """Keep each product's share of its group; take the group's total
    from `parent`, a forecast made at the level `by`."""
    total = aggregate(bottom, by)
    factor = parent[total.columns] / total
    drop = [n for n in bottom.columns.names if n not in by]
    groups = bottom.columns.droplevel(drop)
    return bottom * factor[groups].to_numpy()


def top_down(total: pd.Series, shares: pd.DataFrame) -> pd.DataFrame:
    """Each bottom series gets its share of the total forecast."""
    return shares.mul(total, axis=0)

