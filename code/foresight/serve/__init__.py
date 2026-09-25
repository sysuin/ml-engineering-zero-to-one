"""
Foresight in service: the monthly list written where people read it,
and a small API for one contract or one ticket at a time. Chapter 22
writes it.

    renewal.py   one path from rows to a chance, a rank and reasons,
                 for the monthly job and the API alike
    store.py     the scores database: runs, scores, the rows that
                 were scored, forecasts; never the warehouse itself
    batch.py     the monthly job: python -m foresight.serve.batch
    online.py    Chapter 4's columns for a few contracts, quickly
    triage.py    Chapter 20's model, saved and registered
    api.py       the service: uvicorn foresight.serve.api:app

A root is one whole installation of Foresight: artifacts/ (the
registry), data/foresight/ (the scores database and the holdout) and
mlruns/. The project's own folder is the real one. SANDBOX is a
second, for trying things without touching it; the book's listings
work there.
"""
from __future__ import annotations

import os

import pandas as pd

from foresight.config import DATA

SANDBOX = DATA / "sandbox"


def today() -> pd.Timestamp:
    """The morning the service takes it to be: FORESIGHT_ON if it is
    set, otherwise the calendar's. Meridian's records stop at the end
    of 2025, so the book sets it."""
    return pd.Timestamp(os.getenv("FORESIGHT_ON")
                        or pd.Timestamp.today().date())
