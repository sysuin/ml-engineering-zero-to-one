"""
Chapter 10's forest and leaderboard: the forest fits and ranks like any
Foresight model, gives the same answer twice, runs through Chapter 8's
backtest unchanged, and the leaderboard pairs its models on the same
resamples. Made-up tables only: no warehouse needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table

from foresight.evaluate import auc, evaluate
from foresight.leaderboard import page, paired, run
from foresight.models.forest import RenewalForest
from foresight.models.logistic import RenewalRisk


def small(**kw):
    """A quick forest: fewer trees, smaller leaves, for small tables."""
    return RenewalForest(trees=40, min_leaf=10, **kw)


def test_probabilities_one_per_contract():
    rows = table()
    p = small().fit(rows).predict_proba(rows)
    assert p.shape == (len(rows),)
    assert ((p >= 0) & (p <= 1)).all()


def test_ranks_leavers_above_renewers_on_unseen_rows():
    rows = table(cohorts=10, per_cohort=150)
    fit, score = rows.iloc[:1200], rows.iloc[1200:]
    p = small().fit(fit).predict_proba(score)
    assert auc(score.not_renewed, p) > 0.6   # the gap drives the label


def test_the_same_forest_twice():
    rows = table()
    a = small().fit(rows).predict_proba(rows)
    b = small().fit(rows).predict_proba(rows)
    assert np.array_equal(a, b)


def test_a_different_seed_grows_a_different_forest():
    rows = table()
    a = small().fit(rows).predict_proba(rows)
    b = small(seed=1).fit(rows).predict_proba(rows)
    assert not np.array_equal(a, b)


def test_scaling_a_column_changes_nothing():
    rows = table()
    wide = rows.assign(tenure_days=rows.tenure_days * 1000,
                       spend_365=rows.spend_365 * 7)
    a = small().fit(rows).predict_proba(rows)
    b = small().fit(wide).predict_proba(wide)
    assert np.allclose(a, b)


def test_importances_name_every_column_and_sum_to_one():
    imp = small().fit(table()).importances()
    assert len(imp) == 16
    assert imp.sum() == pytest.approx(1.0)
    assert (imp.diff().dropna() <= 0).all()          # largest first


def _shift(rows):
    """Move a made-up table so that its last six cohorts end in the
    second half of 2024, Foresight's validation split."""
    offset = pd.Timestamp("2024-12-31") - rows.end_date.max()
    return rows.assign(end_date=rows.end_date + offset,
                       moment=rows.moment + offset)


def test_runs_through_the_backtest_unchanged():
    ev = evaluate(table=_shift(table(cohorts=24, per_cohort=60)),
                  reps=20, make_model=small)
    assert ev["scored"].model.between(0, 1).all()
    assert ev["scored"].moment.nunique() == 6


@pytest.fixture(scope="module")
def board():
    """Two models on made-up cohorts, the last six in validation."""
    return run({"logistic": RenewalRisk, "forest": small},
               table=_shift(table(cohorts=24, per_cohort=60)), reps=30)


def test_leaderboard_models_share_their_resamples(board):
    rule = [ev["draws"][("precision", "rule")]
            for ev in board["runs"].values()]
    assert rule[0].equals(rule[1])


def test_a_model_minus_itself_is_nothing(board):
    for what in ("precision", "auc"):
        assert paired(board, what, "forest", "forest") == (0, 0, 0)


def test_leaderboard_page_fits_the_book(board):
    text = page(board)
    assert max(len(line) for line in text.splitlines()) <= 68
    for name in ("logistic", "forest", "rule", "random"):
        assert name in text
