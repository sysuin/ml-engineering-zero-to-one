"""
v0.6's monthly lists for past cohorts, as score.py would have made
them on each cohort's mark. Chapter 25 replays them to run the
retention programme through 2024 and 2025.

score.py makes one list, with reasons, after running the training
job's leakage checks. Replaying two years that way takes minutes; the
rolling-origin backtest fits the same model (v0.5's lasso with Platt
chances) on the same outcomes known at each mark, and ranks the same
way: long-tail contracts only, highest chance first, a tie to the
lower contract id.
"""
from __future__ import annotations

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import calibrated
from foresight.evaluate import backtest
from foresight.models.logistic import CALLS
from foresight.slices import key_accounts
from foresight.train import make_model


def scored(first: str, last: str,
           table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every contract in the cohorts ending in [first, last], with
    v0.6's chance of leaving as `chance`."""
    table = pd.read_parquet(TABLE) if table is None else table
    s = backtest(table, first, last, calibrated(make_model()))
    return s.rename(columns={"model": "chance"})


def lists(scored: pd.DataFrame, k: int = CALLS) -> pd.DataFrame:
    """Each cohort's top k long-tail contracts, ranked."""
    tail = scored[~scored.account_id.isin(key_accounts())]
    ranked = tail.sort_values(["moment", "chance", "contract_id"],
                              ascending=[True, False, True])
    ranked = ranked.assign(rank=ranked.groupby("moment").cumcount() + 1)
    return ranked[ranked["rank"] <= k].reset_index(drop=True)
