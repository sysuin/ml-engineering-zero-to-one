"""
Chapter 12's feature library. Every registered feature is checked twice:
against a value worked out by hand for one made-up account, and for
giving the same answer when records dated on or after the mark are
added. A feature without a hand-worked value fails the suite, so a new
definition cannot enter the library untested. Made-up data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table

from foresight.data.build_table import COLUMNS
from foresight.features import definitions  # noqa: F401  registers
from foresight.features.build import build
from foresight.features.encoding import encode, out_of_fold
from foresight.features.registry import REGISTRY, feature, names
from foresight.features.sources import Context, Sources
from foresight.models.featured import (FeaturedBooster, FeaturedLasso,
                                       linear_view)

MARK = pd.Timestamp("2024-06-01")
DAY = pd.Timedelta(days=1)

# Account 1's orders: (age in days before the mark, value). Age 0 is
# the mark itself and must never be counted; age 397 is too old.
ORDERS_1 = [(0, 9999.0), (1, 500.0), (12, 300.0), (52, 200.0),
            (121, 400.0), (138, 100.0), (183, 600.0), (305, 1000.0),
            (397, 50.0)]
TICKETS_1 = [(0, "Quality", "on the mark: too late"),
             (5, "Quality", "MRD-CLE-001 falling apart, second time"),
             (40, "Delivery", "order 12 still hasnt come, mrd-pac-020"),
             (100, "Billing", "invoice query for mrd-saf-014"),
             (200, "Quality", "whole batch is faulty")]


def sources() -> Sources:
    """Account 1 as above; account 2 a key account; account 3 with no
    orders at all; accounts 4-40 ordering steadily, so that Chapter
    18's segmenter has enough accounts to fit its four segments."""
    g = np.random.default_rng(0)
    orders = [(1, MARK - a * DAY, v) for a, v in ORDERS_1]
    for acct in [2] + list(range(4, 41)):
        step = int(g.integers(3, 40))
        days = pd.date_range("2022-01-03", "2024-12-31",
                             freq=f"{step}D")
        orders += [(acct, d, float(g.integers(50, 900))) for d in days]
    orders = pd.DataFrame(orders,
                          columns=["account_id", "day", "value"])
    # Every order of account 1 is Cleaning from Pemberton, except the
    # day before the mark, which is Sanitation from Voss.
    lines = orders[orders.account_id == 1].copy()
    san = lines.day == MARK - DAY
    lines["category"] = np.where(san, "Sanitation", "Cleaning")
    lines["supplier"] = np.where(san, "Voss Industrial",
                                 "Pemberton Mills")
    tickets = pd.DataFrame(
        [(1, MARK - a * DAY, c, None, b) for a, c, b in TICKETS_1],
        columns=["account_id", "day", "category", "sku", "body"])
    accounts = pd.DataFrame({
        "account_id": range(1, 41),
        "since": [pd.Timestamp("2020-06-01")] * 40,
        "is_key_account": [0, 1] + [0] * 38,
        "postcode": ["00000"] * 40})
    return Sources(orders, lines, tickets, accounts)


def keys(src: Sources) -> pd.DataFrame:
    """One contract per account, as Chapter 4's table would hold it."""
    k = pd.DataFrame({"account_id": range(1, 41)})
    k["contract_id"] = k.account_id + 100
    k["moment"] = MARK
    k.loc[k.account_id == 2, "moment"] = pd.Timestamp("2023-06-01")
    k["segment"] = "Small business"
    o = k.merge(src.orders, on="account_id")
    o = o[o.day < o.moment]
    gap = (o.moment - o.day).dt.days.groupby(o.contract_id).min()
    k["days_since_order"] = k.contract_id.map(gap).astype("Int64")
    return k


def monthly_slope() -> float:
    """spend_slope for account 1, worked out another way: a straight
    line through twelve 30-day months of spend, oldest first."""
    spend = np.zeros(12)
    for age, value in ORDERS_1:
        if 1 <= age <= 360:
            spend[11 - (age - 1) // 30] += value
    slope = np.polyfit(np.arange(12), spend, 1)[0]
    return slope / spend.mean()


YEAR = 3100.0                   # account 1's spend, ages 1 to 365
LOG = np.log
EXPECTED = {                    # feature: (contract, value by hand)
    "orders_365": (101, 7),
    "avg_order_365": (101, YEAR / 7),
    "orders_30d": (101, 2),
    "spend_30d": (101, 800.0),
    "spend_90d": (101, 1000.0),
    "tickets_365": (101, 4),
    "order_trend": (101, LOG(4 / 3)),
    "spend_trend": (101, LOG(1100 / 600)),
    "spend_slope": (101, monthly_slope()),
    "quarter_share": (101, 1000 / YEAR),
    "mark_month_sin": (101, 0.5),
    "mark_month_cos": (101, -np.sqrt(3) / 2),
    "log_tenure_years": (101, LOG(5.0)),
    "days_of_history": (102, 151),       # key account: from 2023
    "no_order_record": (103, 1),
    "complaints_90d": (101, 2),
    "complaints_365": (101, 3),
    "chasing_90d": (101, 2),
    "product_tickets_90d": (101, 2),     # one code in lower case
    "share_cleaning": (101, 2600 / YEAR),
    "share_facilities": (101, 0.0),
    "share_packaging": (101, 0.0),
    "share_safety": (101, 0.0),
    "share_sanitation": (101, 500 / YEAR),
    "categories_365": (101, 2),
    "supplier_kestrel": (101, 0.0),
    "supplier_aldridge": (101, 0.0),
    "supplier_pemberton": (101, 2600 / YEAR),
    "supplier_voss": (101, 500 / YEAR),
    "supplier_penhale": (101, 0.0),
    "suppliers_365": (101, 2),
    "small_x_gap": (101, LOG(1 + 1 / 30)),
    "small_x_trend": (101, LOG(4 / 3)),
    "behaviour_drifting": (103, 0.0),    # no orders: no segment
    "behaviour_newcomers": (103, 0.0),
    "behaviour_long_standing": (103, 0.0),
}


@pytest.fixture(scope="module")
def built():
    src = sources()
    return src, build(keys(src), sources=src).set_index("contract_id")


def test_every_feature_has_a_hand_worked_value():
    assert set(EXPECTED) == set(names())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_value_worked_by_hand(built, name):
    _, out = built
    contract, value = EXPECTED[name]
    assert out.loc[contract, name] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_nothing_on_or_after_the_mark_is_read(built, name):
    """Add an order, a line and a ticket for every account on the
    latest mark and the day after: on the mark for most contracts,
    after it for the one marked a year earlier. No feature may change,
    not even one fitted across accounts."""
    src, before = built
    k = keys(src)
    late = pd.concat([k.assign(day=MARK), k.assign(day=MARK + DAY)])
    late = late[["account_id", "day"]]
    more = Sources(
        pd.concat([src.orders, late.assign(value=1e6)]),
        pd.concat([src.lines, late.assign(category="Sanitation",
                                          supplier="Voss Industrial",
                                          value=1e6)]),
        pd.concat([src.tickets, late.assign(
            category="Quality", sku=None,
            body="MRD-CLE-001 still broken, second time")]),
        src.accounts)
    after = build(k, [name], more).set_index("contract_id")
    pd.testing.assert_series_equal(after[name], before[name])


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_nothing_after_a_mark_reaches_that_contract(built, name):
    """Contract 102 is marked on 1 June 2023. Give every account orders
    and tickets through the year after that: its value may not move.
    A segmenter fitted on a later date fails this test."""
    src, before = built
    k = keys(src)
    days = pd.date_range("2023-06-01", "2024-05-31", freq="7D")
    late = pd.DataFrame([(a, d) for a in k.account_id for d in days],
                        columns=["account_id", "day"])
    more = Sources(
        pd.concat([src.orders, late.assign(value=5000.0)],
                  ignore_index=True),
        pd.concat([src.lines, late.assign(category="Safety",
                                          supplier="Kestrel Supply",
                                          value=5000.0)]),
        pd.concat([src.tickets, late.assign(
            category="Delivery", sku=None, body="still waiting")]),
        src.accounts)
    after = build(k, [name], more).set_index("contract_id")
    assert after.loc[102, name] == before.loc[102, name]


def test_one_definition_per_name():
    assert not set(names()) & set(COLUMNS)
    with pytest.raises(ValueError):
        feature("order_trend", "trends")(lambda ctx: None)
    with pytest.raises(ValueError):
        feature("spend_365", "windows")(lambda ctx: None)


def test_every_feature_says_what_it_is():
    for f in REGISTRY.values():
        assert f.about and f.added and f.group
        assert f.window <= 365 and f.linear in ("raw", "log")


def test_a_window_cannot_reach_the_mark():
    src = sources()
    ctx = Context(keys(src), src)
    with pytest.raises(ValueError):
        ctx.window("orders", 0)
    assert (ctx.window("orders", 365).age >= 1).all()


def test_build_gives_one_number_per_contract(built):
    _, out = built
    assert len(out) == 40
    assert not out[names()].isna().any().any()


# ------------------------------------------------ target encoding
def test_encoding_shrinks_small_categories_towards_the_rate():
    cats = pd.Series(["a"] * 4 + ["b"] * 96)
    y = pd.Series([1] * 4 + [0] * 96)
    enc = encode(cats, y, pd.Series(["a", "b", "new"]), weight=4)
    prior = 0.04
    assert enc[0] == pytest.approx((4 + 4 * prior) / 8)
    assert enc[2] == pytest.approx(prior)       # unseen: the rate


def test_out_of_fold_never_uses_a_rows_own_account():
    """One category per account: out of fold, an account's rows can
    only be given the overall rate of the other folds."""
    g = np.random.default_rng(1)
    acct = np.repeat(np.arange(200), 2)
    y = pd.Series((g.random(400) < 0.3).astype(int))
    naive = encode(pd.Series(acct), y, pd.Series(acct))
    oof = out_of_fold(pd.Series(acct), y, pd.Series(acct))
    left = y.to_numpy() == 1
    assert naive[left].mean() > naive[~left].mean() + 0.01
    assert abs(oof[left].mean() - oof[~left].mean()) < 0.01


# ------------------------------------------------ the models
def enriched(**kw):
    t = table(cohorts=12, per_cohort=100, **kw)
    g = np.random.default_rng(2)
    t["order_trend"] = g.normal(0, 1, len(t))
    t["orders_365"] = g.integers(0, 50, len(t)).astype(float)
    return t


def test_linear_view_logs_what_the_library_says_to():
    t = enriched()
    X = linear_view(t, ["order_trend", "orders_365"])
    assert np.allclose(X.orders_365, np.log1p(t.orders_365))
    assert np.allclose(X.order_trend, t.order_trend)


def test_featured_models_fit_and_add_their_columns():
    t = enriched()
    extra = ["order_trend", "orders_365"]
    lasso = FeaturedLasso(extra).fit(t)
    assert set(extra) <= set(lasso.weights().index)
    boost = FeaturedBooster(extra, min_leaf=10, most=200,
                            patience=20).fit(t)
    for m in (lasso, boost):
        p = m.predict_proba(t)
        assert p.shape == (len(t),) and ((p > 0) & (p < 1)).all()
    # the monotone constraint still names the gap, and only the gap
    rising = boost.params(10)["monotone_constraints"]
    assert sum(rising) == 1 and len(rising) == 11 + len(extra)
