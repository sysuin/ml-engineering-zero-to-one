"""
When to retrain, and how a new model earns its place. Chapter 24.

A policy says on which mornings a model is refitted, always on the
outcomes on record that morning and on nothing later:

    never      fitted once, on the day it was registered, then frozen
    monthly    refitted on the morning of every mark
    trigger    refitted on the first mark after each trigger day, and
               frozen in between

shadow() runs contenders beside production. Each scores every cohort
at its mark under its own policy; nothing it says reaches the account
team, and its scores wait, like production's, for the outcomes.

gate() is Chapter 11's rule applied to that shadow record: on the
cohorts listed since go-live whose outcomes are on record by the day,
a challenger replaces the champion only if it is not behind at
capacity and a paired interval, on precision at capacity or on AUC,
clears zero in its favour. The AUC is taken within each cohort and
averaged, because a list is made within a cohort, and Chapter 16
showed that an AUC pooled across months moves when a model's monthly
map moves whole cohorts. forward() applies the gate on the morning of
every mark, before the list is made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from foresight.config import SEED, rng
from foresight.evaluate import (HISTORY_FROM, REPS, auc, hits_at_k,
                                interval)
from foresight.models.logistic import CALLS, days_since_rule
from foresight.monitor.cohorts import GO_LIVE, known

LEAST = 3               # cohorts with outcomes before the gate opens


@dataclass(frozen=True)
class Contender:
    name: str
    make_model: object          # () -> an unfitted model
    policy: str = "monthly"     # "never", "monthly" or "trigger"
    triggers: tuple = ()        # days, for policy "trigger"


def fit_day(c: Contender, mark, start=GO_LIVE) -> pd.Timestamp:
    """The morning whose outcomes the model scoring `mark` learned
    from."""
    mark, start = pd.Timestamp(mark), pd.Timestamp(start)
    if c.policy == "never":
        return start
    if c.policy == "monthly":
        return mark
    fired = [pd.Timestamp(d) for d in c.triggers
             if pd.Timestamp(d) <= mark]
    return max([start, *fired]) if fired else start


def shadow(rows: pd.DataFrame, contenders, start=GO_LIVE
           ) -> pd.DataFrame:
    """Every contender's chance for every contract marked from `start`
    on: one row per contract and contender."""
    history = rows[rows.end_date >= HISTORY_FROM]
    listed = rows[rows.moment >= pd.Timestamp(start)]
    fitted, out = {}, []
    for mark, cohort in listed.groupby("moment"):
        for c in contenders:
            day = fit_day(c, mark, start)
            if (c.name, day) not in fitted:
                fitted[c.name, day] = c.make_model().fit(
                    known(history, day))
            out.append(cohort[["moment", "contract_id", "end_date",
                               "not_renewed"]].assign(
                name=c.name, fitted_on=day,
                chance=fitted[c.name, day].predict_proba(cohort),
                rule=np.asarray(days_since_rule(cohort), float)))
    return pd.concat(out, ignore_index=True)


def record(s: pd.DataFrame, name: str) -> pd.DataFrame:
    """One contender's rows, laid out as evaluate.measure() reads
    them: its chance as "model"."""
    r = s[s.name == name].rename(columns={"chance": "model"})
    return r.assign(base=r.model).reset_index(drop=True)


def _measures(y, a, b, ids, cohorts, k: int = CALLS) -> tuple:
    """a minus b: leavers at capacity per call, and AUC within each
    cohort, averaged over the cohorts."""
    hits = area = 0.0
    for rows in cohorts:
        cy, ci = y[rows], ids[rows]
        hits += (hits_at_k(cy, a[rows], ci, k)
                 - hits_at_k(cy, b[rows], ci, k))
        area += auc(cy, a[rows]) - auc(cy, b[rows])
    return hits / (k * len(cohorts)), area / len(cohorts)


def paired(a: pd.DataFrame, b: pd.DataFrame, reps: int = REPS,
           seed: int = SEED) -> dict:
    """a minus b on the same contracts, with 95% intervals from
    resampling contracts within each cohort (Chapter 8's bootstrap)."""
    assert (a.contract_id.to_numpy() == b.contract_id.to_numpy()).all()
    y, ids = a.not_renewed.to_numpy(), a.contract_id.to_numpy()
    pa, pb = a.model.to_numpy(), b.model.to_numpy()
    cohorts = list(a.groupby("moment").indices.values())
    point = _measures(y, pa, pb, ids, cohorts)
    g, draws = rng(seed), []
    for _ in range(reps):
        drawn = [r[g.integers(0, len(r), len(r))] for r in cohorts]
        draws.append(_measures(y, pa, pb, ids, drawn))
    d = np.array(draws)
    return {"precision": (point[0], *interval(d[:, 0])),
            "auc": (point[1], *interval(d[:, 1]))}


def gate(s: pd.DataFrame, champion: str, challenger: str, day,
         least: int = LEAST) -> dict:
    """Chapter 11's rule on the shadow cohorts known by `day`."""
    a = known(record(s, challenger), day)
    b = known(record(s, champion), day)
    cohorts = a.moment.nunique()
    out = {"day": pd.Timestamp(day), "challenger": challenger,
           "champion": champion, "cohorts": cohorts,
           "promote": False}
    if cohorts < least:
        return out
    diff = paired(a, b)
    out.update(diff)
    ahead = diff["precision"][1] > 0 or diff["auc"][1] > 0
    out["promote"] = diff["precision"][0] >= 0 and ahead
    return out


def forward(s: pd.DataFrame, champion: str, challengers) -> tuple:
    """Walk the marks in order. Each morning, gate every challenger
    against the champion of the day; a promotion takes effect for that
    morning's list. Returns the gate's log and, for every mark, the
    model whose list went to the account team."""
    log, served = [], {}
    challengers = list(challengers)
    for mark in sorted(s.moment.unique()):
        for name in list(challengers):
            g = gate(s, champion, name, mark)
            log.append(g)
            if g["promote"]:
                challengers.remove(name)
                challengers.append(champion)
                champion = name
        served[pd.Timestamp(mark)] = champion
    return pd.DataFrame(log), pd.Series(served, name="served")


def production(s: pd.DataFrame, served: pd.Series) -> pd.DataFrame:
    """The lists the account team received: each cohort scored by the
    model serving on its mark."""
    rows = [record(s, name)[lambda r, m=mark: r.moment == m]
            for mark, name in served.items()]
    return pd.concat(rows, ignore_index=True)
