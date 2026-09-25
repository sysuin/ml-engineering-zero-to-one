"""
Why Foresight gave a contract its score, and what its models lean on
overall. Chapter 16 writes it; score.py puts it to work.

Two questions, two kinds of tool.

Overall. permutation_importance() shuffles one column at a time within
each monthly cohort and measures what the model loses on contracts it
did not learn from. Every repeat draws a fresh resample of the
contracts and a fresh shuffle, so the range it reports covers both
kinds of luck. partial_dependence() sets one column to each value on a
grid for every contract and averages the chances.

One contract. contributions() splits a contract's log-odds into a base,
the model's average over the contracts it learned from, and one part
per table column. The parts add up to the log-odds exactly:

    linear models       weight times standardised value, summed over
                        a category's columns: for a linear model these
                        are its SHAP values
    RenewalBooster      TreeSHAP, which LightGBM computes itself
    Calibrated          the inner model's parts times the slope of its
                        Platt map, which is a line on the log-odds

reasons() turns the largest parts that raise the risk into sentences an
account manager can check against the account. unfamiliar() marks the
values a model has seen too rarely to score with any confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.config import SEED, rng
from foresight.decide import Calibrated, Platt
from foresight.evaluate import (HISTORY_FROM, auc, hits_at_k, known_by,
                                resample)
from foresight.models.boosting import RenewalBooster
from foresight.models.boosting import columns as tree_columns
from foresight.models.logistic import CALLS

COLUMNS = ["days_since_order", "orders_90d", "orders_prev_90d",
           "spend_365", "tickets_90d", "tenure_days", "segment",
           "region", "term_months", "legacy_terms", "discount_pct"]
"""Chapter 4's columns, the ones v0.5's model reads (train.COLUMNS)."""
CATEGORIES = ("segment", "region")
NUMERIC = [c for c in COLUMNS if c not in CATEGORIES]
TOGETHER = {"orders": ("orders_90d", "orders_prev_90d")}
"""Columns a reason speaks about as one: the two order counts say
"fewer orders than last quarter" only together (Chapter 12)."""
SMALLEST = 0.05         # log-odds: a smaller push is not a reason
RARE = 20               # training rows: fewer is too few to trust


# ------------------------------------------------ one contract
def _source(name: str) -> str:
    """The table column a model's column was built from."""
    if name == "log_spend_365":
        return "spend_365"
    return name.split("=")[0]


def _linear(model, rows: pd.DataFrame):
    X = model.columns(rows)
    Z = model.scaler_.transform(X.to_numpy(dtype=float))
    parts = pd.DataFrame(Z * model.model_.coef_[0], index=rows.index,
                         columns=X.columns)
    parts = parts.T.groupby(_source, sort=False).sum().T
    # The scaler was fitted on the training rows, so their average
    # standardised value is 0 and the base is the intercept.
    return np.full(len(rows), float(model.model_.intercept_[0])), parts


def _tree(model, rows: pd.DataFrame):
    X = (model.columns(rows) if hasattr(model, "columns")
         else tree_columns(rows))
    raw = model.model_.predict(X, pred_contrib=True)
    parts = pd.DataFrame(raw[:, :-1], index=rows.index,
                         columns=X.columns)
    return raw[:, -1], parts


def contributions(model, rows: pd.DataFrame):
    """(base, parts): base + parts.sum(axis=1) is each contract's
    log-odds of leaving. One column of parts per table column."""
    if isinstance(model, Calibrated):
        if not isinstance(model.map_, Platt):
            raise TypeError("only a Platt map is a line in log-odds")
        base, parts = contributions(model.model_, rows)
        a, b = model.map_.a_, model.map_.b_
        return a * base + b, parts * a
    if isinstance(model, RenewalBooster):
        return _tree(model, rows)
    if hasattr(model, "scaler_") and hasattr(model, "model_"):
        return _linear(model, rows)
    raise TypeError(f"no contributions for {type(model).__name__}")


def log_odds(p) -> np.ndarray:
    """A chance as log-odds, unclipped."""
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------ reasons
def facts(train: pd.DataFrame) -> dict:
    """What "typical" means in a reason: medians of the training rows,
    and how often each segment and region left in them."""
    out = {c: float(train[c].astype(float).median()) for c in NUMERIC}
    out["all"] = float(train.not_renewed.mean())
    for c in CATEGORIES:
        out[c] = train.groupby(c).not_renewed.mean().to_dict()
    return out


def say(column: str, r, t: dict) -> str:
    """One reason, in words, from a contract's row `r`."""
    if column == "days_since_order":
        if pd.isna(r.days_since_order):
            return "No order on record"
        return (f"Last order {int(r.days_since_order)} days ago"
                f" (typical: {t['days_since_order']:.0f})")
    if column == "orders":
        now, before = int(r.orders_90d), int(r.orders_prev_90d)
        if now == before == 0:
            return "No orders in the last six months"
        if now < before:
            return (f"Orders fell from {before} to {now},"
                    " quarter on quarter")
        return (f"{now} orders in the last 90 days"
                f" (typical: {t['orders_90d']:.0f})")
    if column == "spend_365":
        return (f"Spent ${r.spend_365:,.0f} in the last year"
                f" (typical: ${t['spend_365']:,.0f})")
    if column == "tickets_90d":
        return (f"{int(r.tickets_90d)} support tickets in 90 days"
                f" (typical: {t['tickets_90d']:.0f})")
    if column == "tenure_days":
        return (f"A customer for {r.tenure_days / 365.25:.1f} years"
                f" (typical: {t['tenure_days'] / 365.25:.1f})")
    if column in CATEGORIES:
        name = getattr(r, column)
        rate = t[column].get(name, t["all"])
        where = "the " + name if column == "region" else name
        return (f"{where}: {rate:.1%} of these left,"
                f" {t['all']:.1%} of all")
    if column == "term_months":
        return f"A {int(r.term_months)}-month term"
    if column == "legacy_terms":
        return "Still on legacy contract terms"
    if column == "discount_pct":
        if pd.isna(r.discount_pct):
            return "No discount on record"
        return (f"A discount of {int(r.discount_pct)}%"
                f" (typical: {t['discount_pct']:.0f}%)")
    return column


def reasons(parts: pd.DataFrame, rows: pd.DataFrame, t: dict,
            n: int = 3, smallest: float = SMALLEST
            ) -> pd.Series:
    """Up to n reasons per contract: the parts that raise its risk
    most, largest first, each at least `smallest` in log-odds."""
    parts = parts.copy()
    for name, cols in TOGETHER.items():
        present = [c for c in cols if c in parts]
        if present:
            parts[name] = parts[present].sum(axis=1)
            parts = parts.drop(columns=present)
    out = {}
    for i, r in zip(rows.index, rows.itertuples(index=False)):
        p = parts.loc[i]
        up = p[p >= smallest].sort_values(ascending=False,
                                          kind="stable")
        out[i] = [say(c, r, t) for c in up.index[:n]]
    return pd.Series(out, dtype=object)


def unfamiliar(train: pd.DataFrame, rows: pd.DataFrame,
               columns=NUMERIC, rare: int = RARE) -> pd.DataFrame:
    """True where fewer than `rare` training rows have a value at least
    as far out, on the same side of the median: a value the model has
    rarely or never seen."""
    flags = {}
    for c in columns:
        seen = np.sort(train[c].astype(float).dropna().to_numpy())
        x = rows[c].astype(float).to_numpy()
        high = len(seen) - np.searchsorted(seen, x, side="left")
        low = np.searchsorted(seen, x, side="right")
        n = np.where(x >= np.median(seen), high, low)
        flags[c] = ~np.isnan(x) & (n < rare)
    return pd.DataFrame(flags, index=rows.index)


# ------------------------------------------------ overall
def by_cohort(table: pd.DataFrame, first: str, last: str,
              make_model) -> dict:
    """The model each cohort ending in [first, last] is scored by,
    fitted on the outcomes known at its mark, as backtest() fits it."""
    history = table[table.end_date >= HISTORY_FROM]
    marks = table[table.end_date.between(first, last)].moment.unique()
    return {pd.Timestamp(m): make_model().fit(known_by(history, m))
            for m in np.sort(marks)}


def scorer(models: dict):
    """score(rows): every row scored by its own cohort's model."""
    def score(rows: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(rows))
        for mark, idx in rows.groupby("moment").indices.items():
            out[idx] = models[pd.Timestamp(mark)].predict_proba(
                rows.iloc[idx])
        return out
    return score


def hits(rows: pd.DataFrame, score, k: int = CALLS) -> int:
    """Leavers in the top k of each cohort, ties to the lower id."""
    y, ids = rows.not_renewed.to_numpy(), rows.contract_id.to_numpy()
    score = np.asarray(score, dtype=float)
    return sum(hits_at_k(y[i], score[i], ids[i], k)
               for i in rows.groupby("moment").indices.values())


def shuffled(rows: pd.DataFrame, columns, g) -> pd.DataFrame:
    """`rows` with `columns` shuffled together within each cohort: the
    same permutation for each, so their relation to one another stays
    and their relation to everything else goes."""
    order = np.arange(len(rows))
    for idx in rows.groupby("moment").indices.values():
        order[idx] = idx[g.permutation(len(idx))]
    out = rows.copy()
    for c in columns:
        out[c] = rows[c].take(order).set_axis(out.index)
    return out


def permutation_importance(rows: pd.DataFrame, score, columns,
                           repeats: int = 50, seed: int = SEED,
                           k: int = CALLS) -> pd.DataFrame:
    """For each column, or tuple of columns shuffled together, the fall
    in AUC and the leavers lost from the top k of each cohort when it
    is shuffled: the mean over `repeats`, and the middle 95%."""
    g = rng(seed)
    rows = rows.reset_index(drop=True)
    names = [c if isinstance(c, str) else "+".join(c) for c in columns]
    drops = {n: [] for n in names}
    lost = {n: [] for n in names}
    for _ in range(repeats):
        s = resample(rows, g).reset_index(drop=True)
        y, base = s.not_renewed.to_numpy(), score(s)
        a, h = auc(y, base), hits(s, base, k)
        for name, c in zip(names, columns):
            p = score(shuffled(s, (c,) if isinstance(c, str) else c, g))
            drops[name].append(a - auc(y, p))
            lost[name].append(h - hits(s, p, k))
    out = []
    for n in names:
        d, v = np.array(drops[n]), np.array(lost[n], dtype=float)
        out.append({"column": n, "auc": d.mean(),
                    "auc_lo": np.percentile(d, 2.5),
                    "auc_hi": np.percentile(d, 97.5),
                    "leavers": v.mean(),
                    "leavers_lo": np.percentile(v, 2.5),
                    "leavers_hi": np.percentile(v, 97.5)})
    return pd.DataFrame(out).set_index("column")


def partial_dependence(model, rows: pd.DataFrame, column: str,
                       grid) -> pd.Series:
    """The average chance over `rows` with `column` set to each value
    of `grid` for every contract, everything else as it was."""
    return pd.Series([float(np.mean(model.predict_proba(
        rows.assign(**{column: v})))) for v in grid], index=list(grid))
