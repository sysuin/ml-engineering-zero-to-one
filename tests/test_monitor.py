"""
Chapter 24: the monitor. PSI by hand and with bins that stay fixed,
notices and outcomes counted only once they exist, a price feed that
alerts once on a step and never on noise, a forecast limit that one
bad month cannot inflate, retraining that never learns from an outcome
not yet on record, and a gate that promotes only what the evidence
supports. Made-up tables only; no warehouse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table

from foresight.config import rng
from foresight.monitor import alerts, cohorts, feeds, forecast, policy
from foresight.monitor.psi import Baseline, edges, level, psi, shares
from foresight.monitor.watch import arrived, early, limits, volume


# ------------------------------------------------ psi
def test_psi_by_hand():
    a = pd.Series({"0": 0.5, "1": 0.5})
    b = pd.Series({"0": 0.25, "1": 0.75})
    want = (0.25 - 0.5) * np.log(0.25 / 0.5) \
        + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert psi(a, b) == pytest.approx(want)
    assert psi(a, a) == 0.0


def test_psi_survives_an_empty_bin():
    a = pd.Series({"0": 1.0})
    b = pd.Series({"0": 0.5, "1": 0.5})
    assert np.isfinite(psi(a, b)) and psi(a, b) > 1


def test_bins_come_from_the_reference_and_stay_fixed():
    g = rng(1)
    ref = pd.DataFrame({"x": g.normal(0, 1, 5000)})
    base = Baseline(ref, ["x"])
    cuts = base.cuts["x"].copy()
    moved = pd.DataFrame({"x": g.normal(1, 1, 2000),
                          "moment": pd.Timestamp("2025-01-01")})
    base.table(moved)
    assert (base.cuts["x"] == cuts).all()
    assert len(cuts) == 9
    same = pd.DataFrame({"x": g.normal(0, 1, 2000),
                         "moment": pd.Timestamp("2025-01-01")})
    assert base.column(same, "x") < 0.02
    assert base.column(moved, "x") > 0.25


def test_missing_values_and_categories_have_their_own_bins():
    s = shares(pd.Series([1.0, None, 3.0, None]), edges([1, 2, 3, 4]))
    assert s["missing"] == 0.5
    c = shares(pd.Series(["a", "a", "b", None]))
    assert c["a"] == 0.5 and c["missing"] == 0.25


def test_level_words():
    assert [level(v) for v in (0.05, 0.1, 0.3)] == ["stable", "watch",
                                                    "moved"]


# ------------------------------------------------ what exists when
def _scored(seed=0):
    t = table(cohorts=4, per_cohort=100, seed=seed)
    t["chance"] = 0.08
    t["rule"] = t.days_since_order.astype(float)
    t["notice_date"] = t.end_date.where(t.not_renewed == 1) \
        - pd.Timedelta(days=60)
    return t


def test_notices_count_only_once_every_notice_is_due():
    t = _scored()
    first = t.moment.min()
    assert early(t, first + pd.Timedelta(days=30)).empty
    e = early(t, first + pd.Timedelta(days=31))
    assert len(e) == 1
    c = t[t.moment == first]
    assert e.noticed.iloc[0] == c.not_renewed.sum()
    assert e.expected.iloc[0] == pytest.approx(0.08 * len(c))


def test_outcomes_arrive_the_day_after_the_end():
    t = _scored()
    end = t.end_date.min()
    assert arrived(t, end).empty
    a = arrived(t, end + pd.Timedelta(days=1))
    assert len(a) == 1 and a.contracts.iloc[0] == 100


def test_poisson_limits_bracket_the_mean():
    lo, hi = limits([20.0])
    assert lo[0] < 20 < hi[0]
    assert limits([20.0], alpha=0.1)[1][0] < hi[0]


def test_volume_against_the_marks_before():
    t = _scored()
    v = volume(t, months=2)
    assert v.usual.isna().sum() == 2
    assert v.ratio.dropna().eq(1.0).all()


def test_exposure_is_zero_before_the_rise_and_for_key_accounts():
    rows = pd.DataFrame({
        "moment": pd.to_datetime(["2025-01-30", "2025-03-02",
                                  "2025-03-02"]),
        "key_account": [0, 0, 1], "supplier_voss": [0.4, 0.4, 0.4]})
    rises = pd.DataFrame({
        "effective_on": pd.to_datetime(["2023-07-01", "2025-02-01"]),
        "scope": ["Small business, all lines",
                  "Voss Industrial lines, non-key accounts"]})
    assert list(cohorts.exposure(rows, rises)) == [0.0, 0.4, 0.0]


# ------------------------------------------------ the feed
def _feed(step_on=80, step=0.08, noise=0.005, days=140, seed=0):
    g = rng(seed)
    day = pd.date_range("2024-01-01", periods=days)
    price = 1 + g.normal(0, noise, days)
    price[step_on:] += step
    return pd.DataFrame({"day": day, "supplier": "Voss / Mid-market",
                         "lines": 30, "price": price, "cost": 1.0})


def test_a_step_alerts_once_on_its_first_day():
    a = feeds.check(_feed(), "price")
    assert len(a) == 1
    assert a.day.iloc[0] == pd.Timestamp("2024-01-01") \
        + pd.Timedelta(days=80)


def test_noise_below_the_threshold_never_alerts():
    assert feeds.check(_feed(step=0.0), "price").empty
    assert feeds.check(_feed(), "cost").empty


# ------------------------------------------------ the forecast
def test_one_bad_month_does_not_inflate_the_limit():
    idx = pd.period_range("2024-01", "2025-06", freq="M")
    wape = pd.Series(0.12, index=idx)
    wape["2024-04"] = 0.40                 # a shock a year before
    wape["2025-04"] = 0.20
    months = pd.DataFrame({"wape": wape, "widened": 0.8})
    w = forecast.watch(months, "2025-01")
    assert w.loc["2025-04", "error_alert"]
    assert not w.loc["2025-03", "error_alert"]


def test_recent_widening_forgets_old_misses():
    rows = []
    for origin in pd.period_range("2023-01", "2024-12", freq="M"):
        miss = 0.5 if origin < pd.Period("2024-01", freq="M") else 0.0
        for i in range(300):
            rows.append({"h": 1, "origin": origin, "target": origin + 1,
                         "series": i, "forecast": 100.0, "q10": 90.0,
                         "q90": 110.0, "actual": 110.0 + 100 * miss})
    r = forecast.widen_recent(pd.DataFrame(rows), months=6)
    late = r[r.origin == pd.Period("2024-12", freq="M")]
    assert late.hi_recent.iloc[0] == pytest.approx(110.0)


# ------------------------------------------------ retraining
class Recorder:
    """A model that remembers the latest end date it learned from."""

    def fit(self, rows):
        self.last_ = rows.end_date.max()
        self.rate_ = rows.not_renewed.mean()
        return self

    def predict_proba(self, rows):
        return np.full(len(rows), self.rate_)


def test_fit_days():
    c = policy.Contender("x", Recorder, "trigger", ("2023-06-10",))
    assert policy.fit_day(c, "2023-05-01", "2023-01-01") \
        == pd.Timestamp("2023-01-01")
    assert policy.fit_day(c, "2023-07-01", "2023-01-01") \
        == pd.Timestamp("2023-06-10")
    frozen = policy.Contender("y", Recorder, "never")
    assert policy.fit_day(frozen, "2023-07-01", "2023-01-01") \
        == pd.Timestamp("2023-01-01")


def test_shadow_never_learns_an_outcome_after_the_mark():
    t = table(cohorts=10, per_cohort=60)
    fitted = []

    def make():
        m = Recorder()
        fitted.append(m)
        return m

    start = t.moment.sort_values().unique()[5]
    s = policy.shadow(t.assign(end_date=t.end_date),
                      [policy.Contender("m", make)], start)
    assert s.moment.min() == start
    for m, mark in zip(fitted, sorted(s.moment.unique())):
        assert m.last_ < mark


def test_the_gate_waits_and_refuses_a_copy():
    t = table(cohorts=6, per_cohort=100)
    s = pd.concat([t.assign(name=n, chance=t.days_since_order
                            .astype(float), rule=0.0)
                   for n in ("champ", "copy")])
    early_day = t.end_date.min()
    assert not policy.gate(s, "champ", "copy", early_day)["promote"]
    g = policy.gate(s, "champ", "copy", "2030-01-01")
    assert g["cohorts"] == 6 and not g["promote"]
    assert g["precision"][0] == 0


def test_the_gate_promotes_a_list_that_knows_the_answer():
    t = table(cohorts=6, per_cohort=100)
    y = t.not_renewed.to_numpy(float)
    s = pd.concat([t.assign(name="champ", chance=rng(3).random(len(t)),
                            rule=0.0),
                   t.assign(name="oracle", chance=y + 0.01 *
                            rng(4).random(len(t)), rule=0.0)])
    assert policy.gate(s, "champ", "oracle", "2030-01-01")["promote"]


# ------------------------------------------------ alerts
def test_an_alert_line_fits_the_page():
    a = alerts.Alert(pd.Timestamp("2025-04-02"), "notices",
                     "list 2025-03-02", 41, 38, "expected 21.9")
    assert len(a.line()) <= 66
