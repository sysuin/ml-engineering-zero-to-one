"""
Chapter 17's forecasting code, on small panels whose answers are known,
and the one rule every part of it must keep: nothing after the origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight import config
from foresight.forecast import baselines as b
from foresight.forecast.backtest import backtest, mae, mape, wape
from foresight.forecast.intervals import coverage, pinball, widen
from foresight.forecast.model import backtest_model, lag_rows
from foresight.forecast.reconcile import (aggregate, proportional,
                                          top_down)

MONTHS = pd.period_range("2022-01", "2025-12", freq="M")
SEASON = np.repeat([1.3, 1.0, 0.9, 0.8], 3)


def panel(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Four seasonal series, two categories in two regions."""
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product([["A", "B"], ["North", "South"]],
                                      names=["category", "region"])
    level = np.array([100.0, 200.0, 300.0, 400.0])
    season = np.tile(SEASON, len(MONTHS) // 12)[:, None]
    y = level * season * (1 + noise * rng.standard_normal(
        (len(MONTHS), 4)))
    return pd.DataFrame(y, index=MONTHS, columns=cols)


def test_seasonal_naive_is_the_same_month_last_year():
    p = panel(noise=0.1)
    f = b.seasonal_naive(p.loc[:"2024-06"], 3)
    assert np.allclose(f, p.loc["2023-09"])


def test_seasonal_naive_is_missing_without_a_year():
    f = b.seasonal_naive(panel().loc[:"2022-06"], 1)
    assert f.isna().all()


def test_naive_and_moving_average():
    p = panel(noise=0.1)
    h = p.loc[:"2024-06"]
    assert np.allclose(b.naive(h, 2), p.loc["2024-06"])
    assert np.allclose(b.moving_average(h, 1),
                       p.loc["2024-04":"2024-06"].mean())


def test_seasonal_smoothing_is_exact_on_a_clean_series():
    p = panel()
    for h in (1, 2, 3):
        f = b.seasonal_smoothing(p.loc[:"2024-06"], h)
        assert np.allclose(f, p.loc[pd.Period("2024-06", "M") + h])


def test_the_smoothing_path_matches_the_function_at_every_origin():
    p = panel(noise=0.1)
    path = b.smoothing_path(p.to_numpy(), (1, 2, 3))
    for i in (4, 11, 12, 30):
        for h in (1, 2, 3):
            f = b.seasonal_smoothing(p.iloc[:i + 1], h).to_numpy()
            both = np.isnan(f) & np.isnan(path[h][i])
            assert np.all(both | np.isclose(f, path[h][i]))


def test_the_backtest_hands_over_history_up_to_the_origin_only():
    p = panel()

    def spy(history, h):
        assert history.index[-1] == seen["target"] - h
        return history.iloc[-1]

    seen = {}
    targets = pd.period_range("2024-01", "2024-03", freq="M")
    for t in targets:
        seen["target"] = t
        backtest(p, spy, horizons=(1, 3), targets=[t])


def test_the_scores():
    a, f = np.array([10.0, 100.0]), np.array([5.0, 110.0])
    assert mae(a, f) == 7.5
    assert mape(a, f) == pytest.approx((0.5 + 0.1) / 2)
    assert wape(a, f) == pytest.approx(15 / 110)


def test_a_missing_forecast_is_refused_not_skipped():
    with pytest.raises(ValueError):
        wape([1.0, 2.0], [1.0, np.nan])


def test_lag_features_do_not_move_when_the_future_does():
    p = panel(noise=0.1)
    q = p.copy()
    q.loc["2024-07":] *= 5                   # rewrite the future
    origin = pd.Period("2024-06", freq="M")
    cols = ["lag0", "lag1", "lag2", "mean6", "mean12", "last_year",
            "last_year_step", "smoothed", "scale"]
    a = lag_rows(p).loc[lambda r: r.origin == origin, cols]
    z = lag_rows(q).loc[lambda r: r.origin == origin, cols]
    assert np.allclose(a, z, equal_nan=True)


def test_the_model_backtest_never_trains_on_the_target():
    p = panel(noise=0.05, seed=config.SEED % 1000)
    q = p.copy()
    q.loc["2024-04"] *= 3                    # only the scored month
    t = pd.period_range("2024-04", "2024-04", freq="M")
    a = backtest_model(p, horizons=(1,), targets=t, n_estimators=20)
    z = backtest_model(q, horizons=(1,), targets=t, n_estimators=20)
    assert np.allclose(a.forecast, z.forecast)
    assert not np.allclose(a.actual, z.actual)


def test_proportional_reconciliation_adds_up_and_keeps_shares():
    cols = pd.MultiIndex.from_tuples(
        [("A", "a1", "N"), ("A", "a2", "N"), ("B", "b1", "N")],
        names=["category", "sku", "region"])
    bottom = pd.DataFrame([[10.0, 30.0, 50.0]], columns=cols)
    parent = pd.DataFrame(
        [[60.0, 40.0]], columns=pd.MultiIndex.from_tuples(
            [("A", "N"), ("B", "N")], names=["category", "region"]))
    r = proportional(bottom, parent, ["category", "region"])
    assert r.iloc[0].tolist() == [15.0, 45.0, 40.0]
    assert aggregate(r, []).iloc[0, 0] == 100.0
    assert aggregate(r, ["region"]).iloc[0, 0] == 100.0


def test_top_down_splits_by_shares():
    shares = pd.DataFrame([[0.25, 0.75]], columns=["x", "y"])
    out = top_down(pd.Series([200.0]), shares)
    assert out.iloc[0].tolist() == [50.0, 150.0]


def test_pinball_loss_is_smallest_at_the_quantile():
    y = np.arange(1, 101, dtype=float)
    losses = {q: pinball(y, np.full(100, q), 0.9) for q in (50, 90, 99)}
    assert min(losses, key=losses.get) == 90


def test_widening_learns_only_from_targets_already_known():
    rows = []
    months = pd.period_range("2024-01", "2024-12", freq="M")
    for k, t in enumerate(months):
        miss = 0.0 if k < 6 else 100.0       # misses only in late 2024
        rows.append({"h": 1, "origin": t - 1, "target": t,
                     "actual": 100 + miss, "forecast": 100.0,
                     "q10": 90.0, "q90": 110.0})
    r = widen(pd.DataFrame(rows), least=3)
    july = r[r.target == pd.Period("2024-07", "M")].iloc[0]
    assert july.hi == pytest.approx(110.0 - 0.1 * 100)  # learnt from H1
    assert coverage(r) == 0.5


@pytest.fixture(scope="module")
def units():
    if not config.ML_WAREHOUSE.exists():
        pytest.skip("the Meridian ML dataset is not generated: "
                    "run `make data`")
    from foresight.forecast.series import CATEGORY_REGION, monthly_units
    return monthly_units(CATEGORY_REGION)


def test_sanitation_is_missing_before_its_launch(units):
    s = units[("Sanitation", "Midwest")]
    assert s.loc[:"2024-03"].isna().all()
    assert (s.loc["2024-04":] > 0).all()


def test_the_panel_starts_when_every_account_is_on_record(units):
    assert str(units.index[0]) == "2023-01"
    assert units.shape == (36, 25)
