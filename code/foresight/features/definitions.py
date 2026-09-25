"""
Every feature in Foresight's library, one definition each. Chapter 12
writes them, section by section, and tests/test_features.py checks
every one: by hand on a made-up account, and for never reading a
record dated on or after the mark.

Each function takes a Context and returns one float per contract. It
reads events only through ctx.window(), so the point-in-time rule is
enforced in one place (sources.py) rather than in forty.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from foresight.features.registry import feature
from foresight.features.sources import RECORD_STARTS

CATEGORIES = ["Cleaning", "Facilities", "Packaging", "Safety",
              "Sanitation"]
STRATEGIC = ["Kestrel Supply", "Aldridge & Co", "Pemberton Mills",
             "Voss Industrial", "Penhale Rubber"]
COMPLAINTS = ("Quality", "Delivery")
CHASING = re.compile(r"second time|still|again|any update|chas",
                     re.IGNORECASE)
SKU = re.compile(r"MRD-[A-Z]{3}-\d{3}", re.IGNORECASE)


def _count(ctx, source, days, mask=None):
    e = ctx.window(source, days)
    if mask is not None:
        e = e[mask(e)]
    return ctx.per_contract(e.groupby("contract_id").size())


def _sum(ctx, source, days, mask=None):
    e = ctx.window(source, days)
    if mask is not None:
        e = e[mask(e)]
    return ctx.per_contract(e.groupby("contract_id").value.sum())


def _share(part: pd.Series, whole: pd.Series) -> pd.Series:
    """part / whole, and 0 where there is no whole."""
    return (part / whole.where(whole > 0)).fillna(0.0)


# ------------------------------------------------ recency, frequency,
#                                                  monetary value
@feature("orders_365", "rfm", reads=("orders",), window=365,
         linear="log", about="orders in the 365 days before the mark")
def orders_365(ctx):
    return _count(ctx, "orders", 365)


@feature("avg_order_365", "rfm", reads=("orders",), window=365,
         linear="log",
         about="average order value over those orders; 0 if none")
def avg_order_365(ctx):
    return _share(_sum(ctx, "orders", 365), _count(ctx, "orders", 365))


# ------------------------------------------------ windows
@feature("orders_30d", "windows", reads=("orders",), window=30,
         linear="log", about="orders in the 30 days before the mark")
def orders_30d(ctx):
    return _count(ctx, "orders", 30)


@feature("spend_30d", "windows", reads=("orders",), window=30,
         linear="log", about="spend in the 30 days before the mark")
def spend_30d(ctx):
    return _sum(ctx, "orders", 30)


@feature("spend_90d", "windows", reads=("orders",), window=90,
         linear="log", about="spend in the 90 days before the mark")
def spend_90d(ctx):
    return _sum(ctx, "orders", 90)


@feature("tickets_365", "windows", reads=("tickets",), window=365,
         linear="log", about="tickets in the 365 days before the mark")
def tickets_365(ctx):
    return _count(ctx, "tickets", 365)


# ------------------------------------------------ trends and ratios
@feature("order_trend", "trends", reads=("orders",), window=180,
         about="log of (orders in the last 90 days + 1) over "
               "(orders in the 90 days before + 1)")
def order_trend(ctx):
    now = _count(ctx, "orders", 90)
    before = _count(ctx, "orders", 180) - now
    return np.log((now + 1) / (before + 1))


@feature("spend_trend", "trends", reads=("orders",), window=180,
         about="log of (spend in the last 90 days + $100) over "
               "(spend in the 90 days before + $100)")
def spend_trend(ctx):
    now = _sum(ctx, "orders", 90)
    before = _sum(ctx, "orders", 180) - now
    return np.log((now + 100) / (before + 100))


@feature("spend_slope", "trends", reads=("orders",), window=360,
         about="slope of monthly spend over twelve 30-day months, "
               "as a share of the average month; + is growing")
def spend_slope(ctx):
    e = ctx.window("orders", 360)
    month = (e.age - 1) // 30                    # 0 = the latest
    m = (e.assign(month=month)
          .groupby(["contract_id", "month"]).value.sum()
          .unstack(fill_value=0.0)
          .reindex(columns=range(12), fill_value=0.0))
    t = 11 - np.arange(12)                       # oldest month is 0
    t = t - t.mean()
    slope = m.to_numpy() @ t / (t @ t)
    level = m.to_numpy().mean(axis=1)
    out = pd.Series(slope, index=m.index)
    return ctx.per_contract(_share(out, pd.Series(level, m.index)))


@feature("quarter_share", "trends", reads=("orders",), window=365,
         about="the last 90 days' share of the year's spend; about "
               "0.25 for a steady account, 0 with no spend")
def quarter_share(ctx):
    return _share(_sum(ctx, "orders", 90), _sum(ctx, "orders", 365))


# ------------------------------------------------ time
@feature("mark_month_sin", "time",
         about="the month of the mark, as a position on a circle "
               "(sine); December sits next to January")
def mark_month_sin(ctx):
    m = ctx.column("moment").dt.month
    return np.sin(2 * np.pi * (m - 1) / 12).astype(float)


@feature("mark_month_cos", "time",
         about="the month of the mark on the same circle (cosine)")
def mark_month_cos(ctx):
    m = ctx.column("moment").dt.month
    return np.cos(2 * np.pi * (m - 1) / 12).astype(float)


@feature("log_tenure_years", "time", reads=("accounts",),
         about="log of 1 + years since the account opened")
def log_tenure_years(ctx):
    a = ctx.sources.accounts.set_index("account_id").since
    since = ctx.column("account_id").map(a)
    years = (ctx.column("moment") - since).dt.days / 365.25
    return np.log1p(years.clip(lower=0)).astype(float)


# ------------------------------------------------ how much is known
@feature("days_of_history", "history", reads=("accounts",),
         about="days of the year before the mark that the warehouse's "
               "order record covers, up to 365 (Chapter 5)")
def days_of_history(ctx):
    a = ctx.sources.accounts.set_index("account_id").is_key_account
    start = ctx.column("account_id").map(a).fillna(0).map(RECORD_STARTS)
    days = (ctx.column("moment") - start).dt.days
    return days.clip(lower=0, upper=365).astype(float)


@feature("no_order_record", "history", reads=("orders",),
         about="1 if no order at all is on record before the mark: "
               "the indicator beside days_since_order's fill")
def no_order_record(ctx):
    return ctx.column("days_since_order").isna().astype(float)


# ---------------------------------------- other tables: tickets
@feature("complaints_90d", "tickets", reads=("tickets",), window=90,
         about="tickets filed as Quality or Delivery, last 90 days")
def complaints_90d(ctx):
    return _count(ctx, "tickets", 90,
                  lambda e: e.category.isin(COMPLAINTS))


@feature("complaints_365", "tickets", reads=("tickets",), window=365,
         linear="log",
         about="tickets filed as Quality or Delivery, last 365 days")
def complaints_365(ctx):
    return _count(ctx, "tickets", 365,
                  lambda e: e.category.isin(COMPLAINTS))


@feature("chasing_90d", "tickets", reads=("tickets",), window=90,
         about="tickets whose text chases something already asked "
               "for ('still', 'again', 'any update'), last 90 days")
def chasing_90d(ctx):
    return _count(ctx, "tickets", 90,
                  lambda e: e.body.str.contains(CHASING))


@feature("product_tickets_90d", "tickets", reads=("tickets",),
         window=90,
         about="tickets naming a product, the code read from the body "
               "in any case (Chapter 18), last 90 days")
def product_tickets_90d(ctx):
    return _count(ctx, "tickets", 90,
                  lambda e: e.body.str.contains(SKU))


# ---------------------------------------- other tables: products
def _category_share(category):
    def share(ctx):
        e = ctx.window("lines", 365)
        part = e[e.category == category].groupby("contract_id").value
        return _share(ctx.per_contract(part.sum()),
                      ctx.per_contract(e.groupby("contract_id")
                                        .value.sum()))
    return share


for _c in CATEGORIES:
    feature(f"share_{_c.lower()}", "products", reads=("lines",),
            window=365,
            about=f"{_c}'s share of the year's spend")(
                _category_share(_c))


@feature("categories_365", "products", reads=("lines",), window=365,
         about="how many of the five categories the account bought "
               "from in the year")
def categories_365(ctx):
    e = ctx.window("lines", 365)
    return ctx.per_contract(e.groupby("contract_id").category.nunique())


# ---------------------------------------- other tables: suppliers
def _supplier_share(supplier):
    def share(ctx):
        e = ctx.window("lines", 365)
        part = e[e.supplier == supplier].groupby("contract_id").value
        return _share(ctx.per_contract(part.sum()),
                      ctx.per_contract(e.groupby("contract_id")
                                        .value.sum()))
    return share


for _s in STRATEGIC:
    _slug = _s.split()[0].lower()
    feature(f"supplier_{_slug}", "suppliers", reads=("lines",),
            window=365,
            about=f"{_s}'s share of the year's spend")(
                _supplier_share(_s))


@feature("suppliers_365", "suppliers", reads=("lines",), window=365,
         linear="log",
         about="how many suppliers' products the account bought")
def suppliers_365(ctx):
    e = ctx.window("lines", 365)
    return ctx.per_contract(e.groupby("contract_id").supplier.nunique())


# ------------------------------------------------ interactions
@feature("small_x_gap", "interactions", reads=("orders",), window=365,
         about="small business (1/0) times log(1 + days since the last "
               "order / 30), the gap's extra weight for small accounts")
def small_x_gap(ctx):
    gap = ctx.column("days_since_order").astype(float).fillna(365)
    small = (ctx.column("segment") == "Small business").astype(float)
    return small * np.log1p(gap / 30)


@feature("small_x_trend", "interactions", reads=("orders",),
         window=180,
         about="small business (1/0) times order_trend")
def small_x_trend(ctx):
    small = (ctx.column("segment") == "Small business").astype(float)
    return small * order_trend(ctx)


# ------------------------------------------------ Chapter 18's segments
BEHAVIOURS = ["Drifting", "Newcomers", "Long-standing core"]


def _behaviour(ctx) -> pd.Series:
    """Each contract's behavioural segment on its mark (Chapter 18),
    from a segmenter fitted on that mark's own history, so that no
    order after the mark moves a centre; an account with no order in
    the year has none."""
    if "behaviour" not in ctx._memo:
        from foresight.segments import account_features, fit_named
        orders = ctx.sources.orders.reset_index(drop=True)
        accounts = ctx.sources.accounts.set_index("account_id")[
            ["since"]]
        parts = []
        for moment, k in ctx.keys.groupby("moment"):
            f = account_features(moment, orders, accounts)
            names = fit_named(f).label(f)
            parts.append(k.account_id.map(names).set_axis(
                k.contract_id))
        ctx._memo["behaviour"] = pd.concat(parts).reindex(ctx.index)
    return ctx._memo["behaviour"]


def _behaviour_is(name):
    def flag(ctx):
        return (_behaviour(ctx) == name).astype(float)
    return flag


for _b in BEHAVIOURS:
    feature(f"behaviour_{_b.split()[0].lower().replace('-', '_')}",
            "behaviour",
            reads=("orders", "accounts"), window=365,
            about=f"1 if Chapter 18's segmenter, fitted at the mark, "
                  f"puts the account in {_b}")(_behaviour_is(_b))
