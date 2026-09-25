"""
Chapter 23's behavioural tests: what the business expects of the model
when one input moves and nothing else does. The model is the one the
settings file describes, fitted on the outcomes known on its as_of day,
scoring the latest six cohorts it could have scored.

Invariance: nothing that should not matter changes a chance. A name is
not an input; the order of the rows is not an input; a region spelt
another way must never become a different number (the model refuses
it, and Chapter 5's "region is one of five" refuses it in the table);
and a renamed account must never lose its history without a word.

Direction: a longer gap since the last order never lowers the risk, a
larger discount never raises it, and more complaint tickets never lower
it. The test the business asked for, that more complaints *raise* the
risk, fails on v0.6 and is kept, marked, with the reason: the lasso
gives tickets no weight at all (listing 23/06).
"""
from __future__ import annotations

import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest
from test_build_table import tiny  # noqa: F401  the hand-built warehouse

from foresight import gate
from foresight.data.build_table import TableCheckError, build
from foresight.evaluate import HISTORY_FROM, known_by

pytestmark = pytest.mark.model
TIE = 1e-12             # chances this close are the same chance


@pytest.fixture(scope="module")
def fitted(real_table):
    model, cfg = gate.candidate()
    history = real_table[real_table.end_date >= HISTORY_FROM]
    rows = known_by(history, cfg["data"]["as_of"])
    return model.fit(rows, rows.not_renewed), rows


@pytest.fixture(scope="module")
def cohorts(fitted):
    _, rows = fitted
    latest = sorted(rows.moment.unique())[-6:]
    return rows[rows.moment.isin(latest)].reset_index(drop=True)


def chance(fitted, rows) -> np.ndarray:
    return fitted[0].predict_proba(rows)[:, 1]


# ------------------------------------------------ invariance
def test_an_account_name_changes_nothing(fitted, cohorts):
    a = chance(fitted, cohorts.assign(name="Hartton Cleaning"))
    b = chance(fitted, cohorts.assign(name="Hartton Hygiene Ltd"))
    assert np.array_equal(a, b)


def test_the_order_of_the_rows_changes_nothing(fitted, cohorts):
    shuffled = cohorts.sample(frac=1, random_state=0)
    a = pd.Series(chance(fitted, cohorts), index=cohorts.contract_id)
    b = pd.Series(chance(fitted, shuffled), index=shuffled.contract_id)
    assert np.allclose(a, b.loc[a.index], rtol=0, atol=TIE)


@pytest.mark.parametrize("spelling", ["midwest", "Mid-west",
                                      "Midwest ", "MIDWEST"])
def test_a_region_spelt_another_way_is_refused_not_rescored(
        fitted, cohorts, spelling):
    rows = cohorts[cohorts.region == "Midwest"].assign(region=spelling)
    with pytest.raises(ValueError, match="unknown categor"):
        chance(fitted, rows)


def test_a_renamed_account_never_loses_its_history_quietly(
        tiny, tmp_path):       # noqa: F811
    renamed = tmp_path / "renamed.db"
    shutil.copy(tiny, renamed)
    with sqlite3.connect(renamed) as con:
        con.execute("UPDATE accounts SET name = 'Acme Hygiene'"
                    " WHERE account_id = 1")
    before = build("2024-01-01", "2025-12-31", warehouse=tiny)
    try:
        after = build("2024-01-01", "2025-12-31", warehouse=renamed)
    except TableCheckError:
        return                  # stopped, loudly: allowed
    pd.testing.assert_frame_equal(before, after)


# ------------------------------------------------ direction
def moved(fitted, rows, column, by):
    """Chances before and after `column` moves by `by`, for the rows
    where it has a value."""
    rows = rows[rows[column].notna()]
    after = rows.assign(**{column: rows[column] + by})
    return chance(fitted, rows), chance(fitted, after)


@pytest.mark.parametrize("days", [30, 90, 180])
def test_a_longer_gap_never_lowers_the_risk(fitted, cohorts, days):
    before, after = moved(fitted, cohorts, "days_since_order", days)
    assert (after >= before - TIE).all()


def test_a_larger_discount_never_raises_the_risk(fitted, cohorts):
    before, after = moved(fitted, cohorts, "discount_pct", 5)
    assert (after <= before + TIE).all()


def test_more_complaints_never_lower_the_risk(fitted, cohorts):
    before, after = moved(fitted, cohorts, "tickets_90d", 3)
    assert (after >= before - TIE).all()


@pytest.mark.xfail(strict=True, reason="v0.6's lasso gives tickets_90d"
                   " a weight of exactly zero (listing 23/06); the"
                   " business is right and the model cannot see it")
def test_more_complaints_raise_the_risk(fitted, cohorts):
    before, after = moved(fitted, cohorts, "tickets_90d", 3)
    assert (after > before + TIE).all()
