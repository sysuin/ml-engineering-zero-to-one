"""
Chapter 18's anomaly alerts, on small series built by hand: what the
counts count, when each chart fires, and what an alert carries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.anomaly import job
from foresight.anomaly.alert import (Alert, evidence, false_alerts,
                                     first_alert, normal_days)
from foresight.anomaly.control import control_chart, poisson_limit
from foresight.anomaly.counts import daily_counts, sku_in
from foresight.anomaly.forest import (c, fit_forest, path_lengths,
                                      rolling_ranks, series_days)


def test_a_product_code_is_found_in_any_case():
    assert sku_in("hi, mrd-cle-001 is falling apart") == "MRD-CLE-001"
    assert sku_in("nothing named here") is None


def test_daily_counts_fill_quiet_days_with_zero():
    t = pd.DataFrame({"day": pd.to_datetime(["2024-01-01",
                                             "2024-01-03"]),
                      "supplier": ["A", "A"]})
    counts = daily_counts(t, "supplier", "2024-01-01", "2024-01-04")
    assert counts["A"].tolist() == [1, 0, 1, 0]


def series(values, start="2024-01-01"):
    days = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"s": values}, index=days)


def test_no_limit_until_the_baseline_is_full():
    ch = control_chart(series([0] * 10 + [50]), window=56)
    assert not ch.alerts.any().any()
    assert ch.upper.isna().all().all()


def test_the_sigma_chart_fires_on_one_ticket_after_a_silent_window():
    ch = control_chart(series([0] * 56 + [1]), window=56,
                       method="sigma", k=10)
    assert ch.alerts.iloc[-1, 0]            # sd of zeros is zero


def test_the_poisson_chart_does_not():
    ch = control_chart(series([0] * 56 + [1, 20]), window=56)
    assert not ch.alerts.iloc[-2, 0]
    assert ch.alerts.iloc[-1, 0]


def test_the_poisson_limit_holds_its_tail_probability():
    from scipy.stats import poisson
    for mean in (0.2, 1.0, 5.0):
        limit = poisson_limit(mean, 0.001, 56)
        assert poisson.sf(limit, mean) <= 0.001
        assert poisson.sf(limit - 1, mean) > 0.001


def test_a_silent_series_is_read_as_one_ticket_in_the_window():
    assert poisson_limit(0.0, 0.001, 56) == poisson_limit(1 / 56, 0.001,
                                                          56)


def test_keeping_alerts_out_keeps_a_long_spike_alerting():
    rng = np.random.default_rng(0)
    values = list(rng.poisson(0.5, 120)) + [12] * 60
    plain = control_chart(series(values), window=56)
    kept = control_chart(series(values), window=56, skip_alerts=True)
    spike = slice(120, 180)
    assert kept.alerts.iloc[spike, 0].all()
    assert plain.alerts.iloc[spike, 0].sum() < 60


def test_the_path_length_of_a_leaf():
    assert c([0, 1, 2]).tolist() == [0.0, 0.0, 1.0]
    assert c(256)[0] == pytest.approx(10.24, abs=0.01)


def test_an_outlier_is_isolated_in_fewer_splits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    model = fit_forest(X, seed=0)
    lengths = path_lengths(model, np.vstack([[[8.0, 8.0]], X[:50]]))
    assert lengths[0] < lengths[1:].min()


def test_path_lengths_agree_with_scikit_learn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    model = fit_forest(X, seed=0)
    ours = 2 ** (-path_lengths(model, X[:20]) / c(256)[0])
    assert np.allclose(ours, -model.score_samples(X[:20]))


def quiet_then_spike(n_days=500, spike_from=450):
    rng = np.random.default_rng(2)
    days = pd.date_range("2023-01-01", periods=n_days, freq="D")
    counts = pd.DataFrame({"a": rng.poisson(1.0, n_days),
                           "b": rng.poisson(3.0, n_days)}, index=days)
    counts.iloc[spike_from:, 0] += 15
    return counts


def test_rolling_ranks_score_each_month_with_an_earlier_year():
    counts = quiet_then_spike()
    ranks = rolling_ranks(series_days(counts), "2024-03-01", seed=0)
    assert ranks.index.get_level_values("day").min() == pd.Timestamp(
        "2024-03-01")
    spike_day = counts.index[450]
    assert ranks.loc[(spike_day, "a"), "rank"] <= 0.01
    assert ranks["rank"].between(0, 1).all()


def test_false_alerts_count_every_series_and_scale_to_a_month():
    days = pd.date_range("2024-01-01", periods=61, freq="D")
    flags = pd.DataFrame(False, index=days, columns=["a", "b"])
    flags.iloc[[5, 40], 0] = True
    flags.iloc[5, 1] = True
    flags.iloc[20, 1] = True              # inside the event
    normal = normal_days(days, "2024-01-01", "2024-03-01",
                         ("2024-01-21", "2024-01-21"))
    n, per_month = false_alerts(flags, normal)
    assert n == 3
    assert per_month == pytest.approx(3 / (60 / 30.4375))


def test_first_alert_on_or_after_the_start():
    flags = series([True, False, False, True])["s"]
    assert first_alert(flags, "2024-01-02") == pd.Timestamp("2024-01-04")
    assert first_alert(flags & False, "2024-01-01") is None


def tiny_tickets():
    """Two years of quiet tickets, then a burst naming one product."""
    rng = np.random.default_rng(3)
    rows = []
    for day in pd.date_range("2022-01-01", "2023-12-31", freq="D"):
        for _ in range(rng.poisson(0.5)):
            rows.append((day, "Quality", "MRD-AAA-001", "Mill",
                         "MRD-AAA-001 is fine"))
        for _ in range(rng.poisson(1.0)):
            rows.append((day, "Billing", None, np.nan, "invoice query"))
    for _ in range(15):
        rows.append((pd.Timestamp("2024-01-01"), "Quality",
                     "MRD-AAA-001", "Mill", "mrd-aaa-001 falls apart"))
    t = pd.DataFrame(rows, columns=["day", "category", "sku", "supplier",
                                    "body"])
    t["ticket_id"] = [f"T{i:05d}" for i in range(len(t))]
    return t


def test_evidence_names_the_product_and_quotes_a_ticket():
    t = tiny_tickets()
    ev = evidence(t, "2024-01-01", "supplier: Mill")
    assert ev[0] == "products named: MRD-AAA-001 x15"
    assert ev[1] == '"mrd-aaa-001 falls apart"'


def test_the_daily_job_raises_one_alert_per_unusual_series():
    t = tiny_tickets()
    alerts = job.run("2024-01-01", t)
    series_hit = {a.series for a in alerts}
    assert series_hit == {"supplier: Mill", "category: Quality"}
    mill = next(a for a in alerts if a.series == "supplier: Mill")
    assert mill.count == 15 and "Poisson" in mill.method
    assert isinstance(mill, Alert) and mill.evidence
    assert all(len(line) <= 68 for a in alerts for line in a.lines())


def test_the_daily_job_is_quiet_on_an_ordinary_day():
    t = tiny_tickets()
    assert job.run("2023-11-15", t[t.day < "2024-01-01"]) == []
