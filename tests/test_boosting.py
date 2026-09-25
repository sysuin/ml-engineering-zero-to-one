"""
Chapter 11's booster: it fits and ranks like any Foresight model, gives
the same answer twice, stops early on months it may see, never lets a
longer gap lower a risk, keeps gaps as gaps, and runs through Chapter
8's backtest and the leaderboard unchanged. Made-up tables only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table
from test_forest import _shift

from foresight.evaluate import auc, evaluate
from foresight.leaderboard import run
from foresight.models.boosting import (RenewalBooster, columns,
                                       watch_split)
from foresight.models.logistic import RenewalRisk


def small(**kw):
    """A quick booster for small tables."""
    return RenewalBooster(min_leaf=10, most=300, patience=30, **kw)


def rows(**kw):
    return table(cohorts=12, per_cohort=100, **kw)


def test_probabilities_one_per_contract():
    t = rows()
    p = small().fit(t).predict_proba(t)
    assert p.shape == (len(t),)
    assert ((p > 0) & (p < 1)).all()


def test_ranks_leavers_above_renewers_on_unseen_rows():
    t = rows()
    fit, score = t.iloc[:1000], t.iloc[1000:]
    p = small().fit(fit).predict_proba(score)
    assert auc(score.not_renewed, p) > 0.6


def test_the_same_booster_twice():
    t = rows()
    a = small().fit(t).predict_proba(t)
    b = small().fit(t).predict_proba(t)
    assert np.array_equal(a, b)


def test_watches_the_last_three_cohorts_and_fits_on_known_outcomes():
    t = rows()
    fit, watch = watch_split(t)
    assert watch.moment.nunique() == 3
    assert watch.moment.min() == np.sort(t.moment.unique())[-3]
    assert (fit.end_date < watch.moment.min()).all()   # on record
    assert fit.index.intersection(watch.index).empty


def test_rounds_come_from_early_stopping():
    m = small().fit(rows())
    assert 1 <= m.rounds_ <= 300
    assert m.rounds_ == int(np.argmin(m.watched_)) + 1
    assert m.model_.n_estimators == m.rounds_


def test_a_longer_gap_never_lowers_the_risk():
    t = rows()
    m = small(leaves=4).fit(t)
    sample = t.iloc[:50]
    risk = [m.predict_proba(sample.assign(days_since_order=g))
            for g in range(0, 400, 20)]
    assert (np.diff(np.array(risk), axis=0) >= -1e-12).all()


def test_gaps_stay_gaps_and_categories_stay_categories():
    t = rows()
    X = columns(t)
    assert X.discount_pct.isna().sum() == int(t.legacy_terms.sum())
    assert isinstance(X.segment.dtype, pd.CategoricalDtype)
    odd = t.iloc[:5].assign(segment=None, discount_pct=np.nan,
                            days_since_order=np.nan)
    p = small().fit(t).predict_proba(odd)
    assert np.isfinite(p).all()


def test_too_little_history_is_refused():
    with pytest.raises(ValueError):
        small().fit(table(cohorts=3, per_cohort=50))


def test_runs_through_the_backtest_and_the_leaderboard():
    t = _shift(table(cohorts=24, per_cohort=60))
    ev = evaluate(table=t, reps=20, make_model=small)
    assert ev["scored"].model.between(0, 1).all()
    board = run({"logistic": RenewalRisk, "boosting": small},
                table=t, reps=20)
    rule = [e["draws"][("precision", "rule")]
            for e in board["runs"].values()]
    assert rule[0].equals(rule[1])
