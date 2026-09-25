"""
Renewal risk from rows to what a person reads: a chance, a place on
the list, and up to three reasons. Chapter 22 writes it, so that the
monthly job and the API answer by one path.

    training_rows(manifest)    the rows the production model learned
                               from, rebuilt and checked against the
                               manifest's hash (reasons need them)
    known(manifest, on)        every outcome on record on the morning
    recalibrate(model, ...)    Platt's map learned again on the latest
                               cohorts the model never saw; the
                               weights stay as a person promoted them
    contributions(model, rows) Chapter 16's parts for the pipeline's
                               model: weight times standardised value,
                               on Platt's line
    explain(...)               chance, reasons, rarely seen values
    score(...)                 a cohort: names, chances, the long tail
                               ranked, key accounts apart
    rule(rows)                 Chapter 3's rule, for when the model
                               cannot answer
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.decide import COSTS, HELD_MONTHS, Platt
from foresight.evaluate import known_by
from foresight.explain import SMALLEST, facts, reasons, unfamiliar
from foresight.models.logistic import CALLS
from foresight.pipeline import features
from foresight.pipeline.artifact import ArtifactError, check_inputs
from foresight.pipeline.model import INPUTS
from foresight.score import accounts
from foresight.tracking import rows_hash


def training_rows(manifest: dict, warehouse: Path = ML_WAREHOUSE
                  ) -> pd.DataFrame:
    """The rows the model was fitted on, made again from the
    warehouse. Refuses if they are not the rows its manifest hashed."""
    data, span = manifest["data"], manifest["settings"]["data"]
    table = features.labelled(span["first"], span["last"], None,
                              warehouse)
    rows = known_by(table[table.end_date >= data["history_from"]],
                    data["as_of"])
    if rows_hash(rows) != data["rows_sha256"]:
        raise ArtifactError("the rows the model learned from have"
                            " changed since it was fitted")
    return rows


def known(manifest: dict, on, warehouse: Path = ML_WAREHOUSE
          ) -> pd.DataFrame:
    """Every outcome on record on the morning of `on`, from the
    history the model's rows were chosen from."""
    first = manifest["data"]["history_from"]
    table = features.labelled(first, f"{pd.Timestamp(on):%Y-%m-%d}",
                              None, warehouse)
    return known_by(table, on)


def recalibrate(model, outcomes: pd.DataFrame, as_of,
                months: int = HELD_MONTHS):
    """(model, cohorts): a copy of `model` whose Platt map is learned
    again from its own scores on the latest `months` cohorts that
    ended on or after `as_of`, so that it never learned from them.
    With fewer such cohorts, the model as fitted and None. The order
    of a list cannot change: the map is a rising line on log-odds."""
    unseen = outcomes[outcomes.end_date >= pd.Timestamp(as_of)]
    marks = np.sort(unseen.moment.unique())
    if len(marks) < months:
        return model, None
    held = unseen[unseen.moment >= marks[-months]]
    scores = model.estimator_.predict_proba(held)[:, 1]
    fresh = copy.copy(model)
    fresh.map_ = Platt().fit(scores, held.not_renewed.to_numpy())
    if fresh.map_.a_ <= 0:              # a map that reverses the order
        return model, None
    return fresh, (pd.Timestamp(marks[-months]),
                   pd.Timestamp(marks[-1]))


def source(name: str) -> str:
    """The table column a model column was made from."""
    for c in INPUTS:
        if name == c or name.startswith(c + "_"):
            return c
    raise KeyError(name)


def contributions(model, rows: pd.DataFrame):
    """(base, parts) for the pipeline's v0.6: base + parts summed is
    each contract's log-odds of leaving, as explain.contributions()
    gives for Chapter 16's model."""
    inner = model.estimator_
    prepare = inner.named_steps["prepare"]
    lasso = inner.named_steps["lasso"].model_
    Z = inner.named_steps["scale"].transform(prepare.transform(rows))
    names = [source(n) for n in prepare.get_feature_names_out()]
    parts = pd.DataFrame(Z * lasso.coef_[0], index=rows.index,
                         columns=names)
    parts = parts.T.groupby(level=0, sort=False).sum().T
    a, b = model.map_.a_, model.map_.b_
    return a * float(lasso.intercept_[0]) + b, parts * a


def explain(model, rows: pd.DataFrame, train: pd.DataFrame,
            typical: dict | None = None) -> pd.DataFrame:
    """`rows` with each contract's chance, reasons, the columns the
    model leans on whose values it has rarely seen, and whether the
    chance is below the break-even."""
    out = rows.copy()
    out["chance"] = model.predict_proba(rows)[:, 1]
    _, parts = contributions(model, rows)
    out["reasons"] = reasons(parts, rows, typical or facts(train))
    odd = unfamiliar(train, rows)
    odd &= parts[odd.columns].abs().to_numpy() >= SMALLEST
    out["unfamiliar"] = [list(odd.columns[f]) for f in odd.to_numpy()]
    out["below"] = out.chance <= COSTS.break_even()
    return out


def rank(scored: pd.DataFrame, k: int = CALLS) -> pd.DataFrame:
    """The long tail ranked by chance, ties to the lower contract id,
    the top k listed; key accounts after, with no chance and no rank
    (Chapter 16)."""
    key = scored.is_key_account == 1
    tail = scored[~key].sort_values(["chance", "contract_id"],
                                    ascending=[False, True])
    tail = tail.assign(rank=np.arange(1, len(tail) + 1))
    keys = scored[key].sort_values("contract_id").assign(
        chance=np.nan, below=False, rank=0,
        reasons=[[] for _ in range(key.sum())],
        unfamiliar=[[] for _ in range(key.sum())])
    out = pd.concat([tail, keys])
    out["listed"] = (out["rank"] >= 1) & (out["rank"] <= k)
    return out.reset_index(drop=True)


def score(model, manifest: dict, rows: pd.DataFrame,
          train: pd.DataFrame, k: int = CALLS,
          warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """One cohort, as the list shows it. Refuses rows whose columns or
    types differ from the ones the model was fitted on."""
    check_inputs(rows, manifest)
    named = accounts(rows, warehouse)
    return rank(explain(model, named, train), k)


def rule(rows: pd.DataFrame, k: int = CALLS) -> pd.DataFrame:
    """Chapter 3's rule in place of the model: the longest gap since
    the last order first, no order last. No chances, one reason."""
    gap = rows.days_since_order
    said = ["No order on record" if pd.isna(g)
            else f"Last order {int(g)} days ago" for g in gap]
    out = rows.assign(chance=gap.astype(float).fillna(-1),
                      reasons=[[s] for s in said],
                      unfamiliar=[[] for _ in said], below=False)
    out = rank(out, k)
    out["chance"] = np.nan
    return out
