"""
The renewal monitor's checks, run forward day by day, with a budget for
attention. Chapter 24 writes it.

Chapter 18 learned that an alert nobody believes is worse than none. So
the checks are split in two. Everything is measured and goes on the
report; only these raise an alert:

    feed        a supplier's price or cost index moved 3% or more
                (feeds.py), one alert per day however many suppliers
    volume      a list with 25% more or fewer contracts than usual
    inputs      a model column with a PSI of 0.25 or more whose drift
                also moved the average score by EFFECT or more in
                log-odds: drift that does not reach the score is
                reported, not alerted
    scores      the PSI of the list's chances at 0.25 or more
    notices     a cohort's notices outside the Poisson limits around
                the leavers its chances expected (watch.early)
    outcomes    the same, once the outcomes are on record

year() replays a stretch of days as the monitor would have lived it,
and returns every alert with the day it would have been raised.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from foresight.explain import contributions
from foresight.monitor.cohorts import notice_due
from foresight.monitor.psi import ACT, Baseline
from foresight.monitor.watch import arrived, early, score_drift, volume

EFFECT = 0.10           # log-odds: a tenth moves a 6% chance by 0.6
VOLUME = 0.25


@dataclass(frozen=True)
class Alert:
    day: pd.Timestamp
    check: str
    subject: str
    value: float
    limit: float
    note: str = ""

    def line(self) -> str:
        return (f"{self.day:%Y-%m-%d}  {self.check:<9}"
                f"{self.subject:<16}{self.value:>7.2f}"
                f"{self.limit:>7.2f} {self.note}")


def effects(model, reference: pd.DataFrame,
            rows: pd.DataFrame) -> pd.Series:
    """How far each column's drift moved the average log-odds: the
    mean of its part (explain.contributions) on `rows` minus its mean
    on the reference rows."""
    _, before = contributions(model, reference)
    _, now = contributions(model, rows)
    return now.mean() - before.mean()


def input_drift(model, reference, cohort, columns) -> pd.DataFrame:
    """PSI and effect on the score of every column, for one cohort, or
    for several (one row per cohort and column)."""
    base = Baseline(reference, columns)
    out = []
    for mark, c in cohort.groupby("moment"):
        moved = effects(model, reference, c)
        for col in columns:
            out.append({"mark": mark, "column": col,
                        "psi": base.column(c, col),
                        "effect": float(moved.get(col, 0.0))})
    return pd.DataFrame(out)


def year(baseline, s: pd.DataFrame, columns, first, last,
         feed: pd.DataFrame | None = None, budget: bool = True,
         acknowledged=(), due: pd.DataFrame | None = None
         ) -> list[Alert]:
    """Every alert the monitor raises from `first` to `last`. `s` holds
    the lists as the team received them (watch.scored). `baseline` is
    (model, rows it learned from), or a function giving that pair for
    a mark: when a new model takes over, drift is measured against
    what the new model learned from. Columns in `acknowledged` are
    measured but never alert. With budget=False, every column over the
    PSI threshold alerts, and every series in the feed on its own: the
    monitor without Chapter 18's discipline. `due` holds every cohort,
    earlier ones included, for the volume check's usual size."""
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    s_all = s if due is None else due
    s = s[s.moment.between(first, last)]
    of = baseline if callable(baseline) else (lambda mark: baseline)
    out: list[Alert] = []

    if feed is not None:
        feed = feed[feed.day.between(first, last)]
        for day, g in feed.groupby("day"):
            if budget:          # one alert a day, however many series
                who = g.supplier.str.split(" / ").str[0]
                out.append(Alert(day, "feed", who.iloc[0][:17],
                                 g.today.iloc[0], g.usual.iloc[0],
                                 f"{g.what.iloc[0]}, {len(g)} series"))
            else:
                out += [Alert(day, "feed", r.supplier[:17], r.today,
                              r.usual, r.what) for r in g.itertuples()]

    n = volume(s_all)
    for mark, r in n[n.index.to_series().between(first, last)
                     ].dropna().iterrows():
        if abs(r.ratio - 1) >= VOLUME:
            out.append(Alert(mark, "volume", "contracts", r.contracts,
                             r.usual))

    for mark, c in s.groupby("moment"):
        model, ref = of(mark)
        cols = list(columns) + [k for k in effects(model, ref, c).index
                                if k not in columns]
        for r in input_drift(model, ref, c, cols).itertuples():
            if r.column in acknowledged and budget:
                continue
            if r.psi >= ACT and (not budget or abs(r.effect) >= EFFECT):
                out.append(Alert(mark, "inputs", r.column, r.psi, ACT,
                                 f"effect {r.effect:+.3f}"))
        d = score_drift(model.predict_proba(ref), c).iloc[0]
        if d.psi >= ACT:
            out.append(Alert(mark, "scores", "chances", d.psi, ACT))

    # Notices and outcomes: checked on the first day each is complete.
    for mark, c in s.groupby("moment"):
        day = notice_due(c).max() + pd.Timedelta(days=1)
        if day <= last:
            e = early(c, day)
            for m, r in e[e.alert].iterrows():
                limit = r.low if r.noticed < r.low else r.high
                out.append(Alert(day, "notices", f"list {m:%Y-%m-%d}",
                                 r.noticed, limit,
                                 f"expected {r.expected:.1f}"))
        day = c.end_date.max() + pd.Timedelta(days=1)
        if day <= last:
            a = arrived(c, day)
            for m, r in a[a.alert].iterrows():
                limit = r.low if r.left < r.low else r.high
                out.append(Alert(day, "outcomes", f"list {m:%Y-%m-%d}",
                                 r.left, limit,
                                 f"expected {r.expected:.1f}"))
    return sorted(out, key=lambda a: (a.day, a.check, a.subject))


def per_month(alerts: list[Alert]) -> pd.DataFrame:
    """Alerts counted by month and check."""
    if not alerts:
        return pd.DataFrame()
    d = pd.DataFrame({"month": [a.day.to_period("M") for a in alerts],
                      "check": [a.check for a in alerts]})
    return d.groupby(["month", "check"]).size().unstack(fill_value=0)


def first(alerts: list[Alert], check: str, since=None):
    """The first alert of a kind, on or after `since`."""
    since = pd.Timestamp(since) if since is not None else None
    for a in alerts:
        if a.check == check and (since is None or a.day >= since):
            return a
    return None

