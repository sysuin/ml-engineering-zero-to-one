"""
The daily anomaly job: run each morning for the day before, it counts
that day's tickets by supplier and by category, checks every series
with the Poisson control chart and the series-day isolation forest,
and prints one alert per unusual series, with its evidence.

    python -m foresight.anomaly.job --day 2024-10-07

The settings are the ones Chapter 18 tuned on 2023 to raise at most
about one false alert a month each, and checked on 2024.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.anomaly.alert import Alert, evidence
from foresight.anomaly.control import control_chart
from foresight.anomaly.counts import all_series, tickets
from foresight.anomaly.forest import fit_forest, series_days
from foresight.config import ML_WAREHOUSE, SEED

HISTORY_FROM = "2022-01-01"
WINDOW = 56             # days of baseline behind every limit
ALPHA = 0.001           # Poisson chart: tail probability to alert at
Q = 0.001               # forest: share of the year that may rank higher
TRAIN_DAYS = 365


def forest_ranks(S: pd.DataFrame, day: pd.Timestamp,
                 seed: int = SEED) -> pd.Series:
    """Each series' rank on `day`, from a forest fitted on the year
    before the start of day's month (as the backtest in Chapter 18)."""
    rows = series_days(S, WINDOW)
    d = rows.index.get_level_values("day")
    month = day.replace(day=1)
    train = rows[(d < month)
                 & (d >= month - pd.Timedelta(days=TRAIN_DAYS))]
    model = fit_forest(train.to_numpy(), seed)
    base = np.sort(-model.score_samples(train.to_numpy()))
    today = rows[d == day]
    score = -model.score_samples(today.to_numpy())
    above = len(base) - np.searchsorted(base, score, side="left")
    return pd.Series(above / len(base),
                     index=today.index.get_level_values("series"))


def run(day, t: pd.DataFrame | None = None,
        warehouse: Path = ML_WAREHOUSE) -> list[Alert]:
    """The alerts for one day, most tickets first."""
    day = pd.Timestamp(day)
    t = tickets(warehouse) if t is None else t
    S = all_series(t, HISTORY_FROM, day.strftime("%Y-%m-%d"))
    chart = control_chart(S, WINDOW, method="poisson", alpha=ALPHA,
                          skip_alerts=True)
    ranks = forest_ranks(S, day)
    alerts = []
    for series in S.columns:
        methods = []
        if chart.alerts.loc[day, series]:
            methods.append("Poisson")
        rank = ranks.get(series, np.nan)
        if rank <= Q:
            methods.append("forest")
        if not methods:
            continue
        alerts.append(Alert(
            day, series, " + ".join(methods), int(S.loc[day, series]),
            float(chart.centre.loc[day, series]),
            limit=float(chart.upper.loc[day, series]),
            rank=float(rank), evidence=evidence(t, day, series)))
    return sorted(alerts, key=lambda a: -a.count)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    args = ap.parse_args(argv)
    alerts = run(args.day)
    if not alerts:
        print(f"{args.day}: no alerts")
    for a in alerts:
        print("\n".join(a.lines()))


if __name__ == "__main__":
    main()
