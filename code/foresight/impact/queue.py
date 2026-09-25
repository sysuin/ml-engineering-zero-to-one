"""
How long a ticket waits for its first read, by the order the queue is
read in. Chapter 25 writes it, to put triage's value in hours.

The desks' reading is replayed on the tickets' real arrival times. A
Desk reads during its hours, one ticket at a time, and each takes the
same number of minutes. Whenever it is free it opens the waiting
ticket with the highest score; a tie goes to the ticket that arrived
first, so a constant score is the queue in order of arrival. The
desk's hours and pace are assumptions, stated wherever they are
printed.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Desk:
    opens: float = 7.0      # the hour reading starts, every day
    closes: float = 19.0    # the hour no new ticket is opened
    minutes: float = 40.0   # minutes one ticket takes, first read on


def _hours(times) -> np.ndarray:
    t = pd.to_datetime(pd.Series(times)).to_numpy("datetime64[s]")
    return t.astype("int64") / 3600.0


def _open(t: float, desk: Desk) -> float:
    """The first moment at or after t the desk is reading."""
    day, hour = divmod(t, 24.0)
    if hour < desk.opens:
        return day * 24 + desk.opens
    if hour >= desk.closes:
        return (day + 1) * 24 + desk.opens
    return t


def first_read(opened_at, score, desk: Desk = Desk()) -> np.ndarray:
    """Hours each ticket waited, from arrival to its first read."""
    arrive = _hours(opened_at)
    order = np.argsort(arrive, kind="stable")
    score = np.asarray(score, dtype=float)
    start = np.full(len(arrive), np.nan)
    waiting: list = []
    t, i, n = arrive[order[0]], 0, len(arrive)
    while i < n or waiting:
        t = _open(t, desk)
        while i < n and arrive[order[i]] <= t:
            k = order[i]
            heapq.heappush(waiting, (-score[k], arrive[k], k))
            i += 1
        if not waiting:
            t = arrive[order[i]]
            continue
        _, _, k = heapq.heappop(waiting)
        start[k] = t
        t += desk.minutes / 60.0
    return start - arrive
