"""
Foresight v0.3's evaluation from Chapter 8: the measures against answers
worked out another way, the backtest against the rule that no model may
learn from an outcome not yet on record, and the bootstrap against what
resampling must preserve. Made-up tables only: no warehouse needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from foresight.config import rng
from foresight.evaluate import (auc, backtest, bootstrap, brier,
                                difference, hits_at_k, interval,
                                known_by, measure, page, report,
                                resample, verdict)


def table(cohorts=8, per_cohort=80, seed=0) -> pd.DataFrame:
    """A renewals table with Chapter 4's columns, monthly cohorts whose
    contracts end on month ends, and a label that a long gap since the
    last order makes more likely."""
    g = rng(seed)
    ends = pd.date_range("2023-01-31", periods=cohorts, freq="ME")
    n = cohorts * per_cohort
    end = np.repeat(ends, per_cohort)
    gap = g.integers(1, 200, n).astype(float)
    z = -3.5 + 0.02 * gap + g.normal(0, 0.5, n)
    legacy = (g.random(n) < 0.2).astype(int)
    return pd.DataFrame({
        "contract_id": np.arange(n) + 1,
        "account_id": np.arange(n) + 1000,
        "moment": end - pd.Timedelta(days=90),
        "end_date": end,
        "days_since_order": pd.array(gap, dtype="Float64").astype("Int64"),
        "orders_90d": g.integers(0, 20, n),
        "orders_prev_90d": g.integers(0, 20, n),
        "spend_365": g.lognormal(9, 1, n),
        "tickets_90d": g.integers(0, 3, n),
        "tenure_days": g.integers(100, 4000, n),
        "segment": g.choice(["Small business", "Mid-market", "Enterprise",
                             "Public sector"], n),
        "region": g.choice(["Midwest", "Northeast", "Southeast",
                            "Southwest", "West"], n),
        "term_months": g.choice([12, 24], n),
        "legacy_terms": legacy,
        "discount_pct": pd.array(np.where(legacy == 1, np.nan,
                                          g.choice([0, 5, 10], n)),
                                 dtype="Float64").astype("Int64"),
        "not_renewed": (g.random(n) < 1 / (1 + np.exp(-z))).astype(int),
    })


def scored(seed=0) -> pd.DataFrame:
    """Scores for three cohorts, as backtest() returns them."""
    g = rng(seed)
    n = 300
    y = (g.random(n) < 0.1).astype(int)
    return pd.DataFrame({
        "moment": np.repeat(pd.date_range("2024-05-02", periods=3,
                                          freq="MS"), 100),
        "contract_id": np.arange(n),
        "not_renewed": y,
        "model": np.clip(0.1 + 0.3 * y + g.normal(0, 0.15, n), 0.01, 0.99),
        "rule": (100 * y + g.integers(0, 150, n)).astype(float),
        "base": 0.1,
    })


def test_auc_matches_the_library_with_and_without_ties():
    g = rng(1)
    y = (g.random(500) < 0.2).astype(int)
    smooth = g.random(500) + y * 0.3
    tied = np.round(smooth * 4)                   # a handful of values
    for s in (smooth, tied):
        assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5         # a tie counts a half


def test_brier_by_hand():
    assert brier([1, 0], [0.8, 0.4]) == pytest.approx((0.04 + 0.16) / 2)


def test_hits_at_k_breaks_ties_by_the_lower_id():
    y = np.array([0, 1, 1, 0])
    score = np.array([5.0, 5.0, 1.0, 9.0])
    ids = np.array([1, 2, 3, 4])
    assert hits_at_k(y, score, ids, k=2) == 0     # ids 4 and 1
    assert hits_at_k(y, score, ids, k=3) == 1     # then id 2


def test_known_by_keeps_only_contracts_ended_before_the_day():
    t = table()
    kept = known_by(t, "2023-03-31")
    assert (kept.end_date < pd.Timestamp("2023-03-31")).all()
    assert len(kept) == 2 * 80                    # January and February


def test_the_backtest_never_learns_an_outcome_not_yet_on_record():
    seen = []

    class Spy:
        def fit(self, rows):
            seen.append(rows.end_date.max())
            self.rate = rows.not_renewed.mean()
            return self

        def predict_proba(self, rows):
            return np.full(len(rows), self.rate)

    t = table()
    out = backtest(t, "2023-06-01", "2023-08-31", make_model=Spy)
    marks = sorted(out.moment.unique())
    assert len(seen) == len(marks) == 3
    for mark, latest in zip(marks, seen):
        assert latest < mark                      # ended before the mark
    assert len(out) == 3 * 80
    sizes = out.groupby("moment").trained_on.first()
    assert (sizes.diff().dropna() > 0).all()      # the history grows


def test_the_backtest_scores_the_rule_on_the_same_rows():
    t = table()
    out = backtest(t, "2023-06-01", "2023-08-31")
    expected = out.days_since_order.astype(float).fillna(-1)
    np.testing.assert_array_equal(out.rule, expected)
    assert out.model.between(0, 1).all()


def test_measure_knows_its_baselines():
    s = scored()
    m = measure(s, k=10)
    rates = s.groupby("moment").not_renewed.mean()
    assert m["precision"]["random"] == pytest.approx(rates.mean())
    best = s.groupby("moment").not_renewed.sum().clip(upper=10).sum()
    assert m["precision"]["perfect"] == pytest.approx(best / 30)
    perfect = s.assign(model=s.not_renewed.astype(float))
    assert measure(perfect, k=10)["precision"]["model"] == \
        pytest.approx(m["precision"]["perfect"])
    assert m["auc"]["random"] == 0.5


def test_resampling_keeps_every_cohort_its_size():
    s = scored()
    r = resample(s, rng(3))
    pd.testing.assert_series_equal(r.groupby("moment").size(),
                                   s.groupby("moment").size())
    for mark, c in r.groupby("moment"):           # no cohort borrows
        assert set(c.contract_id) <= set(
            s[s.moment == mark].contract_id)


def test_the_bootstrap_repeats_with_its_seed_and_brackets_the_point():
    s = scored()
    a, b = bootstrap(s, k=10, reps=200), bootstrap(s, k=10, reps=200)
    pd.testing.assert_frame_equal(a, b)
    lo, hi = interval(a[("auc", "model")])
    assert lo < measure(s, k=10)["auc"]["model"] < hi


def test_a_list_compared_with_itself_differs_by_exactly_nothing():
    s = scored()
    same = s.assign(rule=s.model)
    d = difference(bootstrap(same, k=10, reps=100), "precision")
    assert (d == 0).all()
    ev = {"draws": bootstrap(same, k=10, reps=100)}
    assert "cannot be told apart" in verdict(ev)


def test_the_page_fits_the_page_and_is_written(tmp_path):
    s = scored()
    ev = {"split": "validation", "k": 10, "scored": s.assign(
              trained_on=1000), "point": measure(s, k=10),
          "draws": bootstrap(s, k=10, reps=100)}
    text = page(ev)
    assert max(len(line) for line in text.splitlines()) <= 68
    assert "Model minus rule, paired" in text
    path = tmp_path / "evaluation.md"
    assert report(ev, path) == text
    assert path.read_text().startswith("# Foresight evaluation")
