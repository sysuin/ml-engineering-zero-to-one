"""
The orders feed: what Foresight relies on in `orders` and
`order_lines`, agreed with the order platform team, who load both
tables every night. Chapter 23 writes it.

Every renewal column but the contract's own terms is counted from
these rows (Chapter 4), the monthly job refuses a warehouse whose
orders stop before the mark (Chapter 22), and the forecast sums their
units (Chapter 17). The clauses are the facts those chapters found and
relied on; the owner has agreed to tell Foresight's team before any of
them changes.
"""
from __future__ import annotations

from foresight.contracts.clauses import Column, Contract, Rule
from foresight.data.build_table import LEGACY_FROM

DAY = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
SKU = "MRD-[A-Z][A-Z][A-Z]-[0-9][0-9][0-9]"
CHANNELS = ("Field sales", "Partner", "Telesales", "Web")
DISCOUNTS = (0, 3, 5, 8, 10, 12, 15)    # the price list (Chapter 5)
MIGRATION = "2024-03-01"                # the legacy CRM's last day + 1

ORDERS = Contract(
    table="orders",
    owner="order platform team",
    version="1.0",
    columns=(
        Column("order_id", "integer", low=1),
        Column("account_id", "integer", refers="accounts.account_id",
               why="every window is counted per account"),
        Column("order_date", "text", form=DAY,
               why="a day, never a time: windows end the day before"
                   " the mark"),
        Column("channel", "text", values=CHANNELS),
    ),
    key=("order_id",),
    when="order_date",
    fresh_days=1,
    daily_rows=100,
    rules=(
        Rule("no order under a legacy id after the migration",
             "SELECT COUNT(*) FROM {rows} WHERE account_id >="
             f" {LEGACY_FROM} AND order_date >= '{MIGRATION}'"),
        Rule("every order has at least one line",
             "SELECT COUNT(*) FROM {rows} o WHERE NOT EXISTS (SELECT 1"
             " FROM order_lines l WHERE l.order_id = o.order_id)"),
    ),
    words=(
        "a correction is a new order, never an edit to an old one",
        "a day's orders are all loaded by 06:00 the next morning",
    ),
    readers=("build_table: days_since_order, orders_90d,"
             " orders_prev_90d, spend_365", "serve.batch: fresh()",
             "forecast: monthly units"),
)

ORDER_LINES = Contract(
    table="order_lines",
    owner="order platform team",
    version="1.0",
    columns=(
        Column("order_id", "integer", refers="orders.order_id"),
        Column("line_no", "integer", low=1),
        Column("sku", "text", form=SKU, refers="products.sku"),
        Column("qty", "integer", low=1),
        Column("unit_price", "real", low=0.01,
               why="spend_365 is qty times unit_price"),
        Column("unit_cost", "real", low=0.01),
        Column("discount_pct", "integer", values=DISCOUNTS),
    ),
    key=("order_id", "line_no"),
    words=("prices are what the customer paid, after any discount",),
    readers=("build_table: spend_365", "forecast: units by product"),
)
