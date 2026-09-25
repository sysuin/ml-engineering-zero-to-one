"""
The tickets feed: what Foresight relies on in `tickets`, agreed with
the support desks, whose tool writes a row the moment a ticket is
opened. Chapter 23 writes it.

The renewal model counts tickets in the 90 days before the mark
(Chapter 4), the anomaly job counts them by day and supplier
(Chapter 18), and triage reads the text and learns from the desks'
labels (Chapter 20). The API refuses a ticket over 2,000 characters,
so the feed promises none are stored.
"""
from __future__ import annotations

from foresight.contracts.clauses import Column, Contract, Rule
from foresight.contracts.orders import MIGRATION, SKU
from foresight.data.build_table import LEGACY_FROM

TICKET = "FT-[0-9][0-9][0-9][0-9][0-9][0-9]"
MOMENT = ("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
          " [0-9][0-9]:[0-9][0-9]:[0-9][0-9]")
CATEGORIES = ("Account", "Billing", "Delivery", "Quality", "Returns",
              "Stock")
PRIORITIES = ("Urgent", "High", "Normal", "Low")

TICKETS = Contract(
    table="tickets",
    owner="support desks",
    version="1.0",
    columns=(
        Column("ticket_id", "text", form=TICKET),
        Column("account_id", "integer", refers="accounts.account_id",
               why="tickets_90d is counted per account"),
        Column("opened_at", "text", form=MOMENT,
               why="the day it was opened decides its window"),
        Column("channel", "text", values=("Email", "Phone",
                                          "Web form")),
        Column("sku", "text", required=False, form=SKU,
               refers="products.sku",
               why="the anomaly job counts by supplier"),
        Column("language", "text", values=("en", "de", "es", "fr")),
        Column("body", "text", why="triage reads it"),
        Column("desk", "text", values=("North desk", "South desk")),
        Column("category", "text", values=CATEGORIES,
               why="triage learns from it; Quality and Delivery"
                   " are complaints"),
        Column("priority", "text", values=PRIORITIES),
    ),
    key=("ticket_id",),
    when="opened_at",
    fresh_days=1,
    daily_rows=1,
    rules=(
        Rule("a body has some text and at most 2,000 characters",
             "SELECT COUNT(*) FROM {rows} WHERE"
             " length(trim(body)) NOT BETWEEN 1 AND 2000"),
        Rule("no ticket under a legacy id after the migration",
             "SELECT COUNT(*) FROM {rows} WHERE account_id >="
             f" {LEGACY_FROM} AND opened_at >= '{MIGRATION}'"),
    ),
    words=(
        "a ticket is written when it is opened; its opened_at is never"
        " changed",
        "a desk may correct a category or priority later: the new"
        " value overwrites the old, and the day it changed is not"
        " kept",
    ),
    readers=("build_table: tickets_90d", "anomaly: daily counts",
             "triage: body, category, priority"),
)
