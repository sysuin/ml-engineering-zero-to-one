"""
An alert, the evidence that goes with it, and the two numbers that
judge a detector: how soon it caught a known event, and how often it
cried wolf on ordinary days.

An alert that says only "supplier: Pemberton Mills, 12 tickets" sends
someone to the warehouse to find out why. One that also names the
product most of the tickets mention, and quotes two of them, lets the
quality team act the same morning.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MONTH_DAYS = 30.4375            # 365.25 / 12


@dataclass(frozen=True)
class Alert:
    day: pd.Timestamp
    series: str
    method: str
    count: int
    expected: float
    limit: float | None = None      # control charts
    rank: float | None = None       # isolation forest
    evidence: tuple[str, ...] = ()

    def lines(self, width: int = 64) -> list[str]:
        """The alert as it would be posted, one line per fact."""
        facts = [f"{self.count} tickets",
                 f"expected {self.expected:.1f}"]
        if self.limit is not None:
            facts.append(f"limit {self.limit:.0f}")
        if self.rank is not None:
            facts.append(f"forest rank {self.rank:.4f}")
        out = [f"ALERT {self.day:%Y-%m-%d}  {self.series}  "
               f"[{self.method}]", clip(f"  {', '.join(facts)}", width)]
        return out + [clip(f"  {e}", width) for e in self.evidence]


def clip(s: str, width: int) -> str:
    """Shorten a line to `width` at a word boundary, marking the cut."""
    if len(s) <= width:
        return s
    return s[:width - 3].rsplit(" ", 1)[0] + "..."


def in_series(t: pd.DataFrame, series: str) -> pd.Series:
    """Which tickets count towards a series named 'supplier: X' or
    'category: Y'."""
    by, _, value = series.partition(": ")
    return t[by] == value


def evidence(t: pd.DataFrame, day, series: str,
             quotes: int = 2) -> tuple[str, ...]:
    """What the tickets behind an alert have in common: the products
    they name most often, and the first few bodies."""
    day = pd.Timestamp(day)
    rows = t[(t.day == day) & in_series(t, series)]
    if rows.empty:
        return ()
    top = rows.sku.value_counts().head(2)
    named = ", ".join(f"{sku} x{n}" for sku, n in top.items())
    out = [f"products named: {named or 'none'}"]
    for body in rows.sort_values("ticket_id").body.head(quotes):
        out.append(f'"{body.strip()}"')
    return tuple(out)


def first_alert(flags: pd.Series, start) -> pd.Timestamp | None:
    """The first day on or after `start` that raised an alert."""
    after = flags[flags.index >= pd.Timestamp(start)]
    hits = after[after]
    return hits.index[0] if len(hits) else None


def normal_days(index: pd.DatetimeIndex, first, last,
                event: tuple) -> pd.Series:
    """True on the days from first to last outside the known event."""
    lo, hi = (pd.Timestamp(d) for d in event)
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    inside = (index >= first) & (index <= last)
    return pd.Series(inside & ~((index >= lo) & (index <= hi)),
                     index=index)


def false_alerts(flags, normal: pd.Series) -> tuple[int, float]:
    """Alerts raised on normal days, in total and per month. `flags` is
    a day-by-series frame, or a series indexed by day or (day, series);
    every series flagged on a day counts as one alert."""
    if isinstance(flags, pd.DataFrame):
        keep = normal.reindex(flags.index).to_numpy()
        n = int(flags[keep].sum().sum())
    else:
        days = flags.index.get_level_values(0)
        keep = normal.reindex(days).fillna(False).to_numpy()
        n = int(flags[keep].sum())
    months = normal.sum() / MONTH_DAYS
    return n, n / months
