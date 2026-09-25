"""
Behavioural segments: accounts grouped by how they buy, not by the
label the CRM gave them.

Every feature is measured on a chosen day from the orders dated before
it, under each account's current id, exactly as the training table
measures its own (Chapter 4):

    orders_365     orders in the 365 days before the day
    basket         average order value over those orders ($)
    recency_days   days since the last order (1 = the day before)
    tenure_years   years since the account opened
    trend          log of spend in the last 182 days over the 183
                   before, each with $100 added so a quiet half-year
                   does not divide by zero

An account is in the population if it ordered at least once in the
year. K-means sees five columns built from these: the logs of orders,
basket and recency, tenure, and trend, each standardised, because on
raw units the column with the widest range decides the segments on
its own.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from foresight.config import ML_WAREHOUSE, SEED
from foresight.data.build_table import events, legacy_ids

SEGMENT_DATE = "2024-04-01"     # the day the segments are fitted on
MODEL_COLUMNS = ["log_orders", "log_basket", "log_recency",
                 "tenure_years", "trend"]
RAW_COLUMNS = ["orders_365", "basket", "recency_days", "tenure_years",
               "trend"]


def load(warehouse: Path = ML_WAREHOUSE):
    """Orders under current ids, and the accounts they belong to."""
    con = sqlite3.connect(warehouse)
    orders, _ = events(con, legacy_ids(con))
    accounts = pd.read_sql_query("""
        SELECT account_id, since, segment AS crm_segment,
               is_key_account
        FROM accounts WHERE crm_source = 'Meridian CRM'""", con)
    con.close()
    accounts["since"] = pd.to_datetime(accounts.since)
    return orders, accounts.set_index("account_id")


def account_features(day, orders: pd.DataFrame,
                     accounts: pd.DataFrame) -> pd.DataFrame:
    """One row per account that ordered in the 365 days before `day`."""
    day = pd.Timestamp(day)
    age = (day - orders.day).dt.days
    year = orders[(age >= 1) & (age <= 365)].assign(age=age)
    by = year.groupby("account_id")
    f = pd.DataFrame({"spend_365": by.value.sum(),
                      "orders_365": by.size(),
                      "recency_days": by.age.min()})
    recent = year[year.age <= 182].groupby("account_id").value.sum()
    before = year[year.age > 182].groupby("account_id").value.sum()
    recent = recent.reindex(f.index, fill_value=0.0)
    before = before.reindex(f.index, fill_value=0.0)
    f["basket"] = f.spend_365 / f.orders_365
    f["trend"] = np.log((recent + 100) / (before + 100))
    f = f.join(accounts, how="inner")
    f["tenure_years"] = (day - f.since).dt.days / 365.25
    return f.drop(columns="since")


def model_matrix(f: pd.DataFrame) -> np.ndarray:
    """The five columns k-means sees, before standardising."""
    return np.column_stack([np.log(f.orders_365), np.log(f.basket),
                            np.log(f.recency_days), f.tenure_years,
                            f.trend])


# ------------------------------------------------ k-means, by hand
def assign(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Each row's nearest centre, by squared Euclidean distance."""
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def update(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Each centre moved to the mean of the rows assigned to it."""
    return np.vstack([X[labels == j].mean(axis=0) for j in range(k)])


def inertia(X, labels, centres) -> float:
    """Total squared distance from every row to its own centre."""
    return float(((X - centres[labels]) ** 2).sum())


def kmeans_numpy(X, centres, iterations: int = 100):
    """Lloyd's algorithm from given starting centres. Returns the final
    centres and labels, and the history of both for drawing."""
    history = [(centres.copy(), assign(X, centres))]
    for _ in range(iterations):
        labels = assign(X, centres)
        new = update(X, labels, len(centres))
        history.append((new.copy(), assign(X, new)))
        if np.allclose(new, centres):
            break
        centres = new
    return centres, assign(X, centres), history


# ------------------------------------------------ the segmenter
@dataclass
class Segmenter:
    """Standardise, then k-means; fitted once, applied to any day."""
    k: int = 4
    names: dict[int, str] = field(default_factory=dict)
    seed: int = SEED

    def fit(self, f: pd.DataFrame) -> "Segmenter":
        X = model_matrix(f)
        self.scaler_ = StandardScaler().fit(X)
        self.kmeans_ = KMeans(self.k, n_init=10,
                              random_state=self.seed).fit(
                                  self.scaler_.transform(X))
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict(
            self.scaler_.transform(model_matrix(f)))

    def label(self, f: pd.DataFrame) -> pd.Series:
        """Segment names, or numbers until they have names."""
        ids = pd.Series(self.predict(f), index=f.index)
        return ids.map(self.names) if self.names else ids


def profile(f: pd.DataFrame, labels) -> pd.DataFrame:
    """Medians of the raw features by segment, with sizes and shares."""
    g = f.assign(segment=np.asarray(labels)).groupby("segment")
    out = g[["spend_365", "orders_365", "basket", "recency_days",
             "tenure_years", "trend"]].median()
    out.insert(0, "accounts", g.size())
    out.insert(1, "share", out.accounts / out.accounts.sum())
    out["spend_share"] = g.spend_365.sum() / f.spend_365.sum()
    return out


def name_segments(prof: pd.DataFrame) -> dict[int, str]:
    """Names for the four segments, given by what their profile shows
    rather than by k-means' arbitrary numbering, so a refit that
    numbers them differently still names them the same way:

        Drifting            the longest median time since an order
        Newcomers           the shortest median tenure
        Long-standing core  of the other two, the longer tenure
        Established core    the remaining one
    """
    if len(prof) != 4:
        raise ValueError("the names are for the four-segment solution")
    left = list(prof.index)
    drifting = prof.loc[left, "recency_days"].idxmax()
    left.remove(drifting)
    new = prof.loc[left, "tenure_years"].idxmin()
    left.remove(new)
    longer = prof.loc[left, "tenure_years"].idxmax()
    left.remove(longer)
    return {drifting: "Drifting", new: "Newcomers",
            longer: "Long-standing core", left[0]: "Established core"}


def fit_named(f: pd.DataFrame, k: int = 4) -> Segmenter:
    """The segmenter Foresight uses: four segments, named."""
    seg = Segmenter(k).fit(f)
    seg.names = name_segments(profile(f, seg.predict(f)))
    return seg
