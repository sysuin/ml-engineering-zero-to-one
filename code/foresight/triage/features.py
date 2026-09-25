"""
Tickets as rows a model can learn from: the table, the time split,
the tokens, and the rubric that makes the two desks' labels agree.

    tickets()          every ticket, oldest first, with desk labels
    split(t)           train, validation and test, by the day opened
    tokens(body)       the words of a ticket, with codes and numbers
                       replaced by what they are
    in_transit(body)   the rubric's test for goods damaged in transit
    relabel(t)         the desks' categories, with the rubric applied
    pemberton(t)       the tickets about the Pemberton cloth's defect
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.anomaly.counts import DEFECT_SKU, sku_in
from foresight.config import ML_WAREHOUSE

PRIORITIES = ["Urgent", "High", "Normal", "Low"]
CATEGORIES = ["Account", "Billing", "Delivery", "Quality", "Returns",
              "Stock"]

# Learn from tickets opened before VALID, choose on those before TEST,
# and report once on the rest (2025).
VALID, TEST = "2024-07-01", "2025-01-01"
DEFECT = ("2024-10-07", "2024-12-20")    # Chapter 18's Pemberton defect


def tickets(warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """Every ticket with its body, desk, language and the desk's two
    labels, oldest first."""
    with sqlite3.connect(warehouse) as con:
        t = pd.read_sql_query("""
            SELECT ticket_id, opened_at, desk, language, body,
                   category, priority
            FROM tickets ORDER BY opened_at, ticket_id""", con)
    t["opened_at"] = pd.to_datetime(t.opened_at)
    t["day"] = t.opened_at.dt.normalize()
    return t


def split(t: pd.DataFrame):
    """Train, validation and test, cut by the day a ticket opened."""
    train = t[t.opened_at < VALID]
    valid = t[(t.opened_at >= VALID) & (t.opened_at < TEST)]
    test = t[t.opened_at >= TEST]
    return (train.reset_index(drop=True), valid.reset_index(drop=True),
            test.reset_index(drop=True))


# ------------------------------------------------------------- tokens
CODES = [
    (re.compile(r"mrd-[a-z]{3}-\d{3}"), " sku "),   # a product code
    (re.compile(r"inv-\d+"), " inv "),              # an invoice
    (re.compile(r"po-\d+"), " po "),                # a purchase order
    (re.compile(r"\d+"), " num "),                  # any other number
]
WORD = re.compile(r"[^\W\d_]+")                 # letters, any script


def tokens(body: str) -> list[str]:
    """Lower-case words. A product code becomes 'sku', an invoice 'inv',
    an order number 'num': what matters is that one was mentioned."""
    text = body.lower()
    for pattern, stand_in in CODES:
        text = pattern.sub(stand_in, text)
    return WORD.findall(text)


# ------------------------------------------------------------- rubric
TRANSIT = re.compile(r"crushed|soaked|in transit|dropped the delivery"
                     r"|split open|torn pack|spilled", re.I)


def in_transit(body: str) -> bool:
    """Goods that arrived damaged: the carrier's problem, not the
    product's. The rubric files these under Delivery at both desks."""
    return bool(TRANSIT.search(body))


def relabel(t: pd.DataFrame) -> pd.Series:
    """The desks' categories with the rubric applied: a ticket about
    goods damaged in transit is Delivery, whoever filed it."""
    return t.category.where(~t.body.map(in_transit), "Delivery")


def pemberton(t: pd.DataFrame) -> pd.Series:
    """True for tickets that name the Pemberton cloth during its defect:
    wording that appears nowhere in the training years."""
    during = t.day.between(*DEFECT)
    return during & (t.body.map(sku_in) == DEFECT_SKU)
