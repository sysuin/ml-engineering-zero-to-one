"""
What the monitor can watch about the renewal model, in the order it
becomes possible. Chapter 24 writes it.

    on the morning of the mark    how many contracts were scored, and
                                  how their scores compare with the
                                  scores the model gave in training
    30 days later                 the notices: leavers tell Meridian
                                  60 days before the end
    90 days later                 the outcomes, and with them the
                                  calibration and the list's hits

Every function takes a day and uses only what was on record before it.
The expected number of leavers is the sum of the chances the model gave
the cohort. A count outside the Poisson limits around it, with Chapter
18's tail probability, is an alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from foresight.decide import calibration_slope
from foresight.evaluate import hits_at_k
from foresight.models.logistic import CALLS, days_since_rule
from foresight.monitor.cohorts import (RECORD_ENDS, known, notice_due,
                                       noticed)
from foresight.monitor.psi import edges, psi, shares

ALPHA = 0.001           # Chapter 18's Poisson chart: tail per side


def scored(model, rows: pd.DataFrame) -> pd.DataFrame:
    """The rows with the model's chance and the rule's score."""
    return rows.assign(chance=model.predict_proba(rows),
                       rule=np.asarray(days_since_rule(rows), float))


def limits(expected, alpha: float = ALPHA) -> tuple:
    """The counts below and above which a Poisson count with this
    mean falls with probability at most alpha each side."""
    lam = np.maximum(np.asarray(expected, dtype=float), 1e-9)
    return poisson.ppf(alpha, lam), poisson.isf(alpha, lam)


def score_drift(reference: np.ndarray, s: pd.DataFrame
                ) -> pd.DataFrame:
    """Each cohort on the morning of its mark: how many contracts,
    their average chance, and the PSI of their chances against the
    chances the model gave the rows it learned from."""
    cuts = edges(reference)
    base = shares(reference, cuts)
    rows = []
    for mark, c in s.groupby("moment"):
        rows.append({"mark": mark, "contracts": len(c),
                     "mean": c.chance.mean(),
                     "psi": psi(base, shares(c.chance, cuts))})
    return pd.DataFrame(rows).set_index("mark")


def volume(s: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """Contracts scored at each mark against the average of the
    `months` marks before it."""
    n = s.groupby("moment").size().rename("contracts").to_frame()
    n["usual"] = n.contracts.shift(1).rolling(months).mean()
    n["ratio"] = n.contracts / n.usual
    return n


def early(s: pd.DataFrame, day) -> pd.DataFrame:
    """Notices against expectation, for every cohort whose notices
    were all due before `day`: a leaver's notice is 30 days after the
    mark, so a missing one is a renewal, near enough."""
    day = pd.Timestamp(day)
    rows = []
    for mark, c in s.groupby("moment"):
        if notice_due(c).max() >= day:
            continue
        if c.end_date.max() > RECORD_ENDS:
            continue            # the warehouse has no notice for it
        lo, hi = limits(c.chance.sum())
        n = int(noticed(c, day).sum())
        rows.append({"mark": mark, "contracts": len(c),
                     "expected": c.chance.sum(), "noticed": n,
                     "low": lo, "high": hi,
                     "alert": n > hi or n < lo})
    return _frame(rows, ["contracts", "expected", "noticed", "low",
                         "high", "alert"])


def arrived(s: pd.DataFrame, day, k: int = CALLS) -> pd.DataFrame:
    """Every cohort whose outcomes were on record by `day`: leavers
    against expectation, and the list's hits beside the rule's."""
    rows = []
    for mark, c in known(s, day).groupby("moment"):
        y, ids = c.not_renewed.to_numpy(), c.contract_id.to_numpy()
        lo, hi = limits(c.chance.sum())
        rows.append({"mark": mark, "contracts": len(c),
                     "expected": c.chance.sum(), "left": int(y.sum()),
                     "low": lo, "high": hi,
                     "alert": y.sum() > hi or y.sum() < lo,
                     "hits": hits_at_k(y, c.chance.to_numpy(), ids, k),
                     "rule": hits_at_k(y, c.rule.to_numpy(), ids, k)})
    return _frame(rows, ["contracts", "expected", "left", "low",
                         "high", "alert", "hits", "rule"])


def _frame(rows: list, columns: list) -> pd.DataFrame:
    """Rows indexed by mark, with the columns even when empty."""
    return pd.DataFrame(rows, columns=["mark", *columns]).set_index(
        "mark")


def slope(s: pd.DataFrame, day) -> tuple[float, int]:
    """The calibration slope over every outcome on record by `day`,
    and how many outcomes that is (NaN before any)."""
    k = known(s, day)
    if k.empty or k.not_renewed.nunique() < 2:
        return float("nan"), len(k)
    return calibration_slope(k.chance, k.not_renewed)[0], len(k)
