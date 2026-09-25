"""
Tuning Foresight's booster on training cohorts, within a stated budget.
Chapter 15 writes it.

    python -m foresight.tune            search, fit the choice, log it

The search never sees a validation cohort. Every setting it tries is
scored by Chapter 8's rolling backtest over the cohorts ending October
2023 to April 2024 (TUNING): each cohort by a model fitted on the
outcomes known at its mark, with the rounds chosen inside that fit by
RenewalBooster's early stopping on the latest months it may see. They
are Chapter 9's tuning cohorts less the first three, which leave the
booster too little history to stop early on. Every outcome among them
was on record before the first validation cohort's mark.

The sampler is Optuna's TPE, seeded, and a search stops at the first
of three limits (Budget): a number of trials; a run of trials with no
improvement larger than Chapter 11's line of noise; or a number of
minutes, a guard for a slow machine that should never bind, because a
search stopped by the clock is not repeatable.

A search's trials are kept in data/cache/tune/, under a hash of
everything they depend on: the table's rows, this module and the ones
it calls, the library versions, the budget, the sampler and its seed.
Change any of them and the search runs again; delete the folder and
every trial is recomputed, and must come out the same.

The chosen setting is fitted on the training period and logged, with
its data hash, code hash and scores, by foresight.tracking.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import CACHE, SEED, rng
from foresight.evaluate import auc, backtest, hits_at_k
from foresight.models.boosting import RenewalBooster
from foresight.models.logistic import CALLS, log_loss
from foresight.tracking import code_hash, rows_hash, sha256

TUNING = ("2023-10-01", "2024-04-30")
TRAINING = ("2023-01-01", "2024-06-30")
NOISE = 0.0003      # Chapter 11: smaller changes in log loss are noise

# name: (low, high, log scale, whole numbers)
SPACE = {"leaves": (2, 64, True, True),
         "min_leaf": (5, 400, True, True),
         "rate": (0.02, 0.3, True, False),
         "columns": (0.5, 1.0, False, False),
         "rows": (0.5, 1.0, False, False),
         "l2": (0.001, 30.0, True, False)}
CHAPTER_11 = {"leaves": 2, "min_leaf": 50, "rate": 0.03,
              "columns": 1.0, "rows": 1.0, "l2": 0.0}


class TunableBooster(RenewalBooster):
    """RenewalBooster with three more of LightGBM's settings open: the
    share of columns and of rows each tree may see, and a penalty on
    the squared leaf values. Chapter 11's defaults leave all three
    where LightGBM puts them."""

    def __init__(self, leaves: int = 2, min_leaf: int = 50,
                 rate: float = 0.03, columns: float = 1.0,
                 rows: float = 1.0, l2: float = 0.0):
        super().__init__(rate=rate, leaves=leaves, min_leaf=min_leaf)
        self.columns_share, self.rows_share, self.l2 = columns, rows, l2

    def params(self, rounds: int) -> dict:
        p = super().params(rounds)
        p.update(colsample_bytree=self.columns_share,
                 subsample=self.rows_share,
                 subsample_freq=int(self.rows_share < 1),
                 reg_lambda=self.l2)
        return p


def booster(settings: dict):
    """A make_model for backtest(): a booster with these settings."""
    return lambda: TunableBooster(**settings)


def score(table: pd.DataFrame, settings: dict | None = None,
          cohorts=TUNING, make_model=None, k: int = CALLS) -> dict:
    """One setting on the tuning cohorts: pooled log loss, AUC and
    leavers in the top k of each cohort, and the scored rows."""
    make = make_model or booster({**CHAPTER_11, **(settings or {})})
    scored = backtest(table, *cohorts, make)
    y, p = scored.not_renewed.to_numpy(), scored.model.to_numpy()
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id, k)
               for _, c in scored.groupby("moment"))
    return {"log loss": log_loss(y, p), "auc": auc(y, p),
            "leavers": hits, "scored": scored}


# ------------------------------------------------ where to look
def draw(n: int, seed: int = SEED, space=SPACE) -> list[dict]:
    """Random search: n settings, each drawn independently, uniformly
    on each range (on the log scale where the range says so)."""
    g, out = rng(seed), []
    for _ in range(n):
        s = {}
        for name, (lo, hi, log, whole) in space.items():
            v = (np.exp(g.uniform(np.log(lo), np.log(hi))) if log
                 else g.uniform(lo, hi))
            s[name] = int(round(v)) if whole else float(v)
        out.append(s)
    return out


def grid(values: dict) -> list[dict]:
    """Grid search: every combination of the listed values."""
    names = list(values)
    return [dict(zip(names, combo))
            for combo in itertools.product(*values.values())]


def suggest(trial, space=SPACE) -> dict:
    """Optuna's version of draw(): the sampler chooses each value."""
    s = {}
    for name, (lo, hi, log, whole) in space.items():
        pick = trial.suggest_int if whole else trial.suggest_float
        s[name] = pick(name, lo, hi, log=log)
    return s


# ------------------------------------------------ the budget
@dataclass(frozen=True)
class Budget:
    """When a search stops: after `trials` trials, or after `patience`
    trials in a row that beat the best so far by no more than NOISE,
    or after `minutes`, whichever comes first."""
    trials: int = 60
    patience: int = 20
    minutes: float = 15.0


BUDGET = Budget()


HERE = Path(__file__).resolve().parent
SOURCES = [HERE / "tune.py", HERE / "evaluate.py",
           HERE / "models" / "boosting.py",
           HERE / "models" / "logistic.py"]


def cache_key(table: pd.DataFrame, **what) -> str:
    """A hash of everything a search's trials depend on."""
    libs = {n: metadata.version(n) for n in ("lightgbm", "optuna")}
    return sha256(json.dumps(
        {"rows": rows_hash(table), "code": code_hash(SOURCES),
         "libraries": libs, "cohorts": TUNING, **what},
        sort_keys=True, default=str).encode())[:16]


def search(table: pd.DataFrame, budget: Budget = BUDGET,
           sampler: str = "tpe", seed: int = SEED,
           space=SPACE, cache: bool = True) -> pd.DataFrame:
    """Search the space on the tuning cohorts. One row a trial: its
    settings, scores, the best log loss so far, and why it stopped."""
    key = cache_key(table, budget=asdict(budget), sampler=sampler,
                    seed=seed, space=space)
    kept = CACHE / "tune" / f"{key}.json"
    if cache and kept.exists():
        saved = json.loads(kept.read_text())
        out = pd.DataFrame(saved["trials"])
        out.attrs["stopped by"] = saved["stopped by"]
        return out
    out = _search(table, budget, sampler, seed, space)
    if cache and out.attrs["stopped by"] != "minutes":
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(
            {"trials": out.to_dict("records"),
             "stopped by": out.attrs["stopped by"]}))
    return out


def _search(table, budget, sampler, seed, space) -> pd.DataFrame:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pick = (optuna.samplers.TPESampler(seed=seed) if sampler == "tpe"
            else optuna.samplers.RandomSampler(seed=seed))
    study = optuna.create_study(direction="minimize", sampler=pick)
    state = {"best": np.inf, "since": 0, "why": "trials"}

    def objective(trial):
        s = score(table, suggest(trial, space))
        trial.set_user_attr("auc", s["auc"])
        trial.set_user_attr("leavers", s["leavers"])
        return s["log loss"]

    def patience(study, trial):
        if trial.value < state["best"] - NOISE:
            state["best"], state["since"] = trial.value, 0
        else:
            state["best"] = min(state["best"], trial.value)
            state["since"] += 1
        if state["since"] >= budget.patience:
            state["why"] = "patience"
            study.stop()

    started = time.monotonic()
    study.optimize(objective, n_trials=budget.trials,
                   timeout=budget.minutes * 60, callbacks=[patience])
    if (state["why"] == "trials" and len(study.trials) < budget.trials
            and time.monotonic() - started >= budget.minutes * 60):
        state["why"] = "minutes"
    rows = [{"trial": t.number + 1, **t.params, "log loss": t.value,
             "auc": t.user_attrs["auc"],
             "leavers": t.user_attrs["leavers"]}
            for t in study.trials]
    out = pd.DataFrame(rows)
    out["best so far"] = out["log loss"].cummin()
    out.attrs["stopped by"] = state["why"]
    return out


def best(trials: pd.DataFrame, space=SPACE) -> dict:
    """The settings of the trial with the lowest log loss."""
    row = trials.loc[trials["log loss"].idxmin()]
    return {n: (int(row[n]) if space[n][3] else float(row[n]))
            for n in space}


# ------------------------------------------------ the command
def main(argv=None) -> None:
    from foresight.data.build_table import TABLE
    from foresight.tracking import MLRUNS, RUN_LOG, track
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trials", type=int, default=BUDGET.trials)
    ap.add_argument("--store", default=str(MLRUNS))
    ap.add_argument("--log", default=str(RUN_LOG))
    args = ap.parse_args(argv)
    table = pd.read_parquet(TABLE)
    budget = Budget(args.trials, BUDGET.patience, BUDGET.minutes)
    trials = search(table, budget)
    chosen = best(trials)
    why = trials.attrs["stopped by"]
    print(f"{len(trials)} trials, stopped by {why}")
    print("Chosen: " + ", ".join(f"{k} {v:g}"
                                 for k, v in chosen.items()))
    s = score(table, chosen)
    rows = table[table.end_date.between(*TRAINING)]
    model = TunableBooster(**chosen).fit(rows)
    record = track("booster, tuned", model, chosen, rows, table,
                   {k: s[k] for k in ("log loss", "auc", "leavers")},
                   evaluated_on=f"cohorts ending {TUNING[0]} to"
                                f" {TUNING[1]}",
                   store=args.store, log=args.log)
    print(f"Logged run {record['run']} to {args.store} and {args.log}")


if __name__ == "__main__":
    main()
