"""
Foresight's demand forecast, end to end: the backtest that judges it,
and the forecast for the three months after the last one on record.

    python -m foresight.forecast.report

The forecast is made for every product in every region by the global
model, with an 80% interval from its two quantile models, widened from
its own backtest record. The median is then reconciled middle-out: the
category x region model sets each category's total, and the products
keep their shares of it, so products, categories, regions and the
total add up. The report prints seasonal naive beside every score.
"""
from __future__ import annotations

import pandas as pd

from foresight.config import DATA
from foresight.forecast.backtest import backtest, wape
from foresight.forecast.baselines import seasonal_naive
from foresight.forecast.intervals import coverage, widen
from foresight.forecast.model import (backtest_model, forecast_from,
                                      training_rows)
from foresight.forecast.reconcile import proportional
from foresight.forecast.series import (CATEGORY_REGION, LAST,
                                       PRODUCT_REGION, established,
                                       monthly_units)

OUTPUT = DATA / "foresight" / "demand_forecast.csv"
QUANTILES = (None, 0.1, 0.9)


def panels(by):
    return monthly_units(by), monthly_units(by, strand="tail")


def wide(rows: pd.DataFrame, column: str, names) -> pd.DataFrame:
    """Forecast rows as a frame: (h, target) down, series across."""
    w = rows.pivot_table(index=["h", "target"], columns="series",
                         values=column)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=names)
    return w


def next_quarter(record: pd.DataFrame | None = None,
                 origin=None) -> pd.DataFrame:
    """Product x region forecasts for the three months after origin
    (default: the last month on record), reconciled and widened.
    `record` is the model's backtest, if it has been run already."""
    origin = pd.Period(origin or LAST, freq="M")
    P, T = panels(PRODUCT_REGION)
    C, TC = panels(CATEGORY_REGION)
    if record is None:
        record = backtest_model(P, (T,), alphas=QUANTILES)
    rows, train = training_rows(P, (T,))
    live = forecast_from(origin, rows, train, QUANTILES)
    live = widen(pd.concat([record, live]))
    live = live[live.origin == origin]
    rows, train = training_rows(C, (TC,))
    parent = wide(forecast_from(origin, rows, train), "forecast",
                  C.columns.names)
    median = wide(live, "forecast", P.columns.names)
    factor = proportional(median, parent, CATEGORY_REGION) / median
    out = []
    for column in ("forecast", "lo", "hi"):
        w = wide(live, column, P.columns.names) * factor
        out.append(w.stack(list(P.columns.names)).rename(column))
    return pd.concat(out, axis=1).reset_index()


def main() -> None:
    P, T = panels(PRODUCT_REGION)
    E = established(P)
    record = backtest_model(P, (T,), alphas=QUANTILES)
    model = widen(record[record.series.isin(E.columns)])
    naive = backtest(E, seasonal_naive)
    print(f"Backtest 2024-2025, product x region, {E.shape[1]} series")
    print(f"  {'':<24}{'h=1':>8}{'h=2':>8}{'h=3':>8}")
    rows = {"seasonal naive, WAPE": [wape(g.actual, g.forecast)
                                     for _, g in naive.groupby("h")],
            "model, WAPE": [wape(g.actual, g.forecast)
                            for _, g in model.groupby("h")],
            "80% interval, raw": [coverage(g)
                                  for _, g in model.groupby("h")]}
    known = model.dropna(subset=["lo"])
    rows["80% interval, widened"] = [coverage(g, "lo", "hi")
                                     for _, g in known.groupby("h")]
    for name, cells in rows.items():
        print(f"  {name:<24}" + "".join(f"{c:>8.1%}" for c in cells))

    forecast = next_quarter(record)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT, index=False)
    # Intervals do not add up: the sum of 240 low ends is far below any
    # plausible total. The total is printed as a median only.
    print(f"\nNext quarter, {len(forecast) // 3} series, "
          f"written to {OUTPUT.relative_to(DATA.parent)}")
    total = forecast.groupby("target").forecast.sum()
    for target, units in total.items():
        print(f"  {target}  {units:>9,.0f} units in all")


if __name__ == "__main__":
    main()
