"""
Chapter 9's lasso and Chapter 11's booster, given the feature library's
columns as well as Chapter 4's. Chapter 12 writes it.

    python -m foresight.models.featured     choose settings, leaderboard

Neither model changes. FeaturedLasso is RegularisedRisk with more
columns: each library feature is added as it is, or as log(1 + x) when
its definition says a linear model should see it on a log scale.
FeaturedBooster is RenewalBooster with more columns: LightGBM reads
them raw. The rows given to either must come from
foresight.features.build, which puts the columns there.

Settings are chosen as Chapters 9 and 11 chose them, on the training
period only: the lasso's strength by the rolling backtest over the
tuning cohorts, the booster's tree and leaf size by the watched
months' log loss.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import SEED, rng
from foresight.evaluate import (REPS, auc, backtest, bootstrap,
                                difference, hits_at_k, interval,
                                measure, resample)
from foresight.features.registry import REGISTRY
from foresight.models.boosting import (NUMERIC, RenewalBooster, columns,
                                       watch_split)
from foresight.models.logistic import CALLS, log_loss
from foresight.models.regularised import (STRENGTHS, TUNE,
                                          RegularisedRisk,
                                          head_to_head)

TRAINING = ("2023-01-01", "2024-06-30")
SHAPES = [(2, 20), (2, 50), (4, 20), (4, 50), (8, 20), (8, 50)]


def linear_view(rows: pd.DataFrame, extra) -> pd.DataFrame:
    """The library's columns as a linear model should see them. A
    column the library does not define is passed through as it is."""
    X = pd.DataFrame(index=rows.index)
    for name in extra:
        x = rows[name].astype(float)
        log = name in REGISTRY and REGISTRY[name].linear == "log"
        X[name] = np.log1p(x) if log else x
    return X


class FeaturedLasso(RegularisedRisk):
    """The lasso on Chapter 7's columns plus `extra` features."""

    def __init__(self, extra=(), strength: float = 0.002,
                 penalty: str = "l1", drop: tuple = ()):
        super().__init__(penalty, strength, drop)
        self.extra = tuple(extra)

    def columns(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([super().columns(rows),
                          linear_view(rows, self.extra)], axis=1)


class FeaturedBooster(RenewalBooster):
    """The booster on Chapter 4's columns plus `extra` features."""

    def __init__(self, extra=(), leaves: int = 2, min_leaf: int = 50,
                 **kw):
        super().__init__(leaves=leaves, min_leaf=min_leaf, **kw)
        self.extra = tuple(extra)

    def columns(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = columns(rows)
        for name in self.extra:
            X[name] = rows[name].astype(float)
        return X

    def params(self, rounds: int) -> dict:
        p = super().params(rounds)
        names = NUMERIC + ["segment", "region", *self.extra]
        p["monotone_constraints"] = [int(c in self.rising)
                                     for c in names]
        return p

    def fit(self, rows: pd.DataFrame):
        fit, watch = watch_split(rows, self.months)
        stop = lgb.early_stopping(self.patience, verbose=False)
        stopper = LGBMClassifier(**self.params(self.most)).fit(
            self.columns(fit), fit.not_renewed,
            eval_X=self.columns(watch), eval_y=watch.not_renewed,
            callbacks=[stop])
        self.rounds_ = int(stopper.best_iteration_)
        self.watched_ = stopper.evals_result_["valid_0"][
            "binary_logloss"]
        self.model_ = LGBMClassifier(**self.params(self.rounds_)).fit(
            self.columns(rows), rows.not_renewed)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.columns(rows))[:, 1]


def lasso(extra=(), strength: float = 0.002):
    """A make_model for backtest()."""
    return lambda: FeaturedLasso(extra, strength)


def booster(extra=(), leaves: int = 2, min_leaf: int = 50):
    """A make_model for backtest()."""
    return lambda: FeaturedBooster(extra, leaves, min_leaf)


# ------------------------------------------------ the training period
def tuning_score(table: pd.DataFrame, make_model,
                 k: int = CALLS) -> dict:
    """A model backtested on Chapter 9's tuning cohorts, which end
    before any validation outcome is known: leavers, AUC, log loss."""
    scored = backtest(table, *TUNE, make_model)
    y, p = scored.not_renewed.to_numpy(), scored.model.to_numpy()
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id, k)
               for _, c in scored.groupby("moment"))
    return {"hits": hits, "auc": auc(y, p), "log loss": log_loss(y, p)}


def watched_loss(table: pd.DataFrame, extra=(), leaves: int = 2,
                 min_leaf: int = 50) -> tuple[float, int]:
    """Chapter 11's measure: the booster's lowest log loss on the
    training period's watched months, and the rounds it took."""
    train = table[table.end_date.between(*TRAINING)]
    m = FeaturedBooster(extra, leaves, min_leaf).fit(train)
    return float(min(m.watched_)), m.rounds_


def tune_lasso(table: pd.DataFrame, extra,
               strengths=STRENGTHS) -> pd.DataFrame:
    """Chapter 9's search, on the tuning cohorts: one row a strength."""
    return pd.DataFrame([{"strength": s,
                          **tuning_score(table, lasso(extra, s))}
                         for s in strengths])


def tune_booster(table: pd.DataFrame, extra,
                 shapes=SHAPES) -> pd.DataFrame:
    """Chapter 11's search, on the watched months: one row a shape."""
    rows = []
    for leaves, min_leaf in shapes:
        loss, rounds = watched_loss(table, extra, leaves, min_leaf)
        rows.append({"leaves": leaves, "min_leaf": min_leaf,
                     "rounds": rounds, "watched": loss})
    return pd.DataFrame(rows)


# ------------------------------------------------ choosing groups
NOISE = 0.0003      # Chapter 11: log-loss changes smaller are noise


def evidence(table: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Each group of features added on its own to v0.4's lasso and to
    the booster, judged on the training period only: the change in the
    lasso's log loss and AUC on the tuning cohorts, and in the
    booster's watched log loss. One row a group."""
    base = tuning_score(table, lasso())
    watched, _ = watched_loss(table)
    rows = []
    for name, extra in groups.items():
        s = tuning_score(table, lasso(extra))
        loss, _ = watched_loss(table, extra)
        rows.append({"group": name, "features": len(extra),
                     "lasso": s["log loss"] - base["log loss"],
                     "lasso auc": s["auc"] - base["auc"],
                     "booster": loss - watched})
    out = pd.DataFrame(rows).set_index("group")
    for model in ("lasso", "booster"):      # the rule, per model
        out[f"keep {model}"] = out[model] < -NOISE
    return out


# ------------------------------------------------ judging a gain
def gain(new: pd.DataFrame, old: pd.DataFrame, reps: int = REPS,
         seed: int = SEED) -> dict:
    """Two backtests of the same cohorts, new minus old: leavers at
    capacity, then precision, AUC and log loss, each with its paired
    95% interval. A falling log loss is a gain."""
    both = head_to_head(new, old)
    point, draws = measure(both), bootstrap(both, reps=reps, seed=seed)
    calls = int(both.groupby("moment").size().clip(upper=CALLS).sum())
    out = {"leavers": round((point["precision"]["model"]
                             - point["precision"]["rule"]) * calls)}
    for what in ("precision", "auc"):
        d = point[what]["model"] - point[what]["rule"]
        out[what] = (d, *interval(difference(draws, what)))
    g, losses = rng(seed), []
    cols = both[["moment", "not_renewed", "model", "rule"]]
    for _ in range(reps):               # the same resamples as above
        s = resample(cols, g)
        y = s.not_renewed.to_numpy()
        losses.append(log_loss(y, s.model.to_numpy())
                      - log_loss(y, s.rule.to_numpy()))
    y = both.not_renewed.to_numpy()
    out["log loss"] = (log_loss(y, both.model.to_numpy())
                       - log_loss(y, both.rule.to_numpy()),
                       *interval(losses))
    return out


def shown(g: dict) -> str:
    """A gain on one line of 54 characters: leavers, then precision in
    points and AUC, each with its 95% interval."""
    d, lo, hi = (v * 100 for v in g["precision"])
    a, alo, ahi = g["auc"]
    return (f"{g['leavers']:>+4}  {d:+.1f} ({lo:+.1f} to {hi:+.1f})"
            f"  {a:+.3f} ({alo:+.3f} to {ahi:+.3f})")


SHOWN = f"{'leavers':>7}  {'points':<21}AUC"     # the heading above
