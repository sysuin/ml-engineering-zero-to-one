"""
From a score to a decision: who to call, what the numbers mean as
chances, and how much revenue the chances put at risk. Chapter 14
writes it.

    python -m foresight.decide       validation cohorts, v0.4's lasso

Three parts, each small.

Thresholds. A list can be cut by the cost matrix (call every contract
whose chance of leaving is above the break-even), by capacity (call the
top k of each monthly cohort), or by both (the top k, but never below
the break-even). The model supplies the chances; the cut belongs to the
business, and every function here takes it as an argument.

Calibration. `Calibrated` wraps any Foresight model. Inside fit() it
holds back the latest monthly cohorts in the rows it is given, fits the
model on the outcomes known before them, and learns how that model's
scores map to what happened in the held-back months: Platt scaling (a
two-number line on the log-odds) or isotonic regression (a staircase).
Then it refits the model on every row and applies the map to its
scores. The map never sees the cohorts it will be judged on, and it is
learned out of time, the way it will be used.

Revenue at risk. Each contract's chance of leaving times the revenue it
would take with it, summed by region, with the range the total could
land in if the chances are right.

The oversampling and SMOTE functions at the end are here so that
Chapter 14's comparison can be rerun and tested. Foresight's training
does not call them.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from foresight.config import SEED, rng
from foresight.costs import CostMatrix
from foresight.models.logistic import CALLS, features

# The brief's figures (Chapter 3): a lost long-tail renewal is worth
# $2,854 of gross profit a year; a call saves one would-be leaver in
# four, and takes an hour and a half at $60 an hour.
COSTS = CostMatrix(value_at_stake=2854, save_rate=0.25, call_hours=1.5,
                   hour_cost=60)
HELD_MONTHS = 3         # the latest cohorts a calibrator learns from


# ------------------------------------------------ thresholds
def by_cost(p, costs: CostMatrix = COSTS) -> np.ndarray:
    """Call where the expected save is worth more than the call."""
    return np.asarray(p, dtype=float) > costs.break_even()


def by_capacity(scored: pd.DataFrame, k: int = CALLS,
                score: str = "model") -> np.ndarray:
    """Call the k highest scores of each monthly cohort; a tie goes to
    the lower contract id, as in evaluate.hits_at_k()."""
    ranked = scored.assign(_s=scored[score].to_numpy()).sort_values(
        ["moment", "_s", "contract_id"], ascending=[True, False, True])
    top = ranked.groupby("moment").head(k).index
    return scored.index.isin(top)


def by_both(scored: pd.DataFrame, k: int = CALLS,
            costs: CostMatrix = COSTS,
            score: str = "model") -> np.ndarray:
    """The top k of each cohort, but none below the break-even."""
    return by_capacity(scored, k, score) & by_cost(scored[score], costs)


def net_value(y, called, costs: CostMatrix = COSTS) -> float:
    """What a list made against calling nobody, from the outcomes."""
    y, called = np.asarray(y), np.asarray(called, dtype=bool)
    tp = int((called & (y == 1)).sum())
    fp = int((called & (y == 0)).sum())
    return costs.net_value(tp, fp)


def expected_value(p, called, costs: CostMatrix = COSTS) -> float:
    """What the same list should make, if the chances are right."""
    p, called = np.asarray(p, dtype=float), np.asarray(called, bool)
    return float((p[called] * costs.save_value
                  - costs.call_cost).sum())


def cost_of_errors(y, called, costs: CostMatrix = COSTS) -> float:
    """Wasted calls and missed saves, in dollars: what a list lost
    against knowing the outcomes in advance."""
    y, called = np.asarray(y), np.asarray(called, dtype=bool)
    fp = int((called & (y == 0)).sum())
    fn = int((~called & (y == 1)).sum())
    err = costs.error_costs()
    return fp * err["fp"] + fn * err["fn"]


def expected_cost_of_errors(p, called,
                            costs: CostMatrix = COSTS) -> float:
    """The same cost, as the model's own chances predict it."""
    p, called = np.asarray(p, dtype=float), np.asarray(called, bool)
    err = costs.error_costs()
    return float(((1 - p[called]) * err["fp"]).sum()
                 + (p[~called] * err["fn"]).sum())


# ------------------------------------------------ calibration
def logit(p, eps: float = 1e-6) -> np.ndarray:
    """Log-odds, with the chance kept a hair inside 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope(p, y) -> tuple[float, float]:
    """Fit outcome on log-odds: (slope, intercept). A slope of 1 and
    an intercept of 0 is a model whose chances need no correcting;
    above 1, too timid; below 1, too bold."""
    m = LogisticRegression(C=np.inf).fit(logit(p)[:, None],
                                         np.asarray(y))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    """Contracts sorted by score and cut into equal-count bins: the
    average chance given and the share that left, bin by bin."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    order = np.argsort(p, kind="stable")
    rows = []
    for i, part in enumerate(np.array_split(order, bins)):
        rows.append({"bin": i + 1, "contracts": len(part),
                     "predicted": float(p[part].mean()),
                     "left": float(y[part].mean()),
                     "leavers": int(y[part].sum())})
    return pd.DataFrame(rows)


class Platt:
    """A line on the log-odds: new log-odds = a * old + b."""

    def fit(self, p, y):
        self.a_, self.b_ = calibration_slope(p, y)
        return self

    def __call__(self, p) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a_ * logit(p) + self.b_)))


class Isotonic:
    """A staircase that never steps down: the share that left among
    contracts scored about this high, forced to rise with the score."""

    def fit(self, p, y):
        self.map_ = IsotonicRegression(y_min=0, y_max=1,
                                       out_of_bounds="clip")
        self.map_.fit(np.asarray(p, dtype=float), np.asarray(y))
        return self

    def __call__(self, p) -> np.ndarray:
        return self.map_.predict(np.asarray(p, dtype=float))


METHODS = {"platt": Platt, "isotonic": Isotonic}


def held_back(rows: pd.DataFrame, months: int = HELD_MONTHS):
    """The latest `months` cohorts, and the rows whose outcomes were on
    record at the first of their marks: what a model fitted on the
    early rows could have been scored on, out of time."""
    marks = np.sort(rows.moment.unique())
    held = rows[rows.moment >= marks[-months]]
    early = rows[rows.end_date < marks[-months]]
    if early.empty or early.not_renewed.nunique() < 2:
        raise ValueError("too little history before the held months")
    return early, held


class Calibrated:
    """Any Foresight model, with its chances corrected out of time."""

    def __init__(self, make_model, method: str = "platt",
                 months: int = HELD_MONTHS):
        self.make_model, self.method, self.months = (make_model, method,
                                                     months)

    def fit(self, rows: pd.DataFrame):
        early, held = held_back(rows, self.months)
        scores = self.make_model().fit(early).predict_proba(held)
        self.map_ = METHODS[self.method]().fit(
            scores, held.not_renewed.to_numpy())
        self.model_ = self.make_model().fit(rows)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.map_(self.model_.predict_proba(rows))


def calibrated(make_model, method: str = "platt"):
    """A make_model for backtest(): the model, calibrated."""
    return lambda: Calibrated(make_model, method)


# ------------------------------------------------ revenue at risk
def value_at_stake(rows: pd.DataFrame) -> np.ndarray:
    """A quarter's revenue at the account's run rate: last year's spend
    scaled to 90 days, Chapter 6's best rule on validation."""
    return rows.spend_365.to_numpy(dtype=float) * 90 / 365


def revenue_at_risk(p, value, groups) -> pd.Series:
    """Chance of leaving times revenue at stake, summed by group."""
    return pd.Series(np.asarray(p) * np.asarray(value)).groupby(
        np.asarray(groups)).sum()


def simulate(p, value, groups, reps: int = 2000,
             seed: int = SEED) -> pd.DataFrame:
    """The revenue that walks, if each contract leaves with the chance
    it was given: one row per simulated half-year, one column per
    group. The middle 95% of a column is the range to expect."""
    g = rng(seed)
    p, value = np.asarray(p, dtype=float), np.asarray(value, float)
    codes, names = pd.factorize(np.asarray(groups), sort=True)
    lost = (g.random((reps, len(p))) < p) * value
    out = np.stack([lost[:, codes == i].sum(axis=1)
                    for i in range(len(names))], axis=1)
    return pd.DataFrame(out, columns=names)


# ------------------------------------------------ resampling, compared
def oversample(rows: pd.DataFrame, g) -> pd.DataFrame:
    """Copy leavers at random until they match the renewers."""
    pos = np.flatnonzero(rows.not_renewed.to_numpy() == 1)
    need = len(rows) - 2 * len(pos)
    copies = pos[g.integers(0, len(pos), need)]
    return pd.concat([rows, rows.iloc[copies]])


NUMERIC = ["days_since_order", "orders_90d", "orders_prev_90d",
           "spend_365", "tickets_90d", "tenure_days", "discount_pct"]


def smote(rows: pd.DataFrame, g, k: int = 5) -> pd.DataFrame:
    """Invent leavers until they match the renewers: each new one lies
    at a random point on the line between a real leaver and one of its
    k nearest leavers, measured on standardised columns. Categories,
    terms and dates come from the first of the pair; a gap stays a
    gap."""
    X = features(rows).to_numpy()
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    pos = np.flatnonzero(rows.not_renewed.to_numpy() == 1)
    d = ((Z[pos, None, :] - Z[None, pos, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1, kind="stable")[:, :k]
    need = len(rows) - 2 * len(pos)
    first = g.integers(0, len(pos), need)
    second = near[first, g.integers(0, k, need)]
    step = g.random(need)
    a = rows.iloc[pos[first]].reset_index(drop=True)
    b = rows.iloc[pos[second]].reset_index(drop=True)
    new = a.copy()
    rows = rows.copy()
    for c in NUMERIC:
        av, bv = a[c].astype(float), b[c].astype(float)
        new[c] = (av + step * (bv - av)).fillna(av).to_numpy()
        rows[c] = rows[c].astype(float)
    return pd.concat([rows, new], ignore_index=True)


def undo_weight(p, ratio: float) -> np.ndarray:
    """Chances from a model that saw leavers `ratio` times as often as
    they happen, put back on the scale of the world: divide the odds
    by the ratio."""
    odds = np.asarray(p, dtype=float) / (1 - np.asarray(p, float))
    odds = odds / ratio
    return odds / (1 + odds)


# ------------------------------------------------ the command
def main(argv=None) -> None:
    from foresight.data.build_table import TABLE
    from foresight.evaluate import SPLITS, backtest, interval
    from foresight.models.regularised import maker
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    table = pd.read_parquet(TABLE)
    s = backtest(table, *SPLITS["validation"],
                 calibrated(maker("l1", 0.002)))
    y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
    print("v0.4's lasso, Platt-calibrated out of time: validation")
    for name, called in (("top 40", by_capacity(s)),
                         ("break-even", by_cost(p)),
                         ("both", by_both(s))):
        print(f"  {name:<12}{called.sum():>5} calls"
              f"{net_value(y, called):>+11,.0f}")
    v = value_at_stake(s)
    risk = revenue_at_risk(p, v, s.region)
    band = simulate(p, v, s.region)
    real = revenue_at_risk(y, v, s.region)
    print(f"\n{'region':<12}{'at risk':>10}{'95% range':>22}"
          f"{'walked':>10}")
    for r in risk.index:
        lo, hi = interval(band[r])
        print(f"{r:<12}{risk[r]:>10,.0f}{lo:>11,.0f} to{hi:>9,.0f}"
              f"{real[r]:>10,.0f}")


if __name__ == "__main__":
    main()
