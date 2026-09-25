# The timestamp audit: every column's source, and the deletion test on
# three training cohorts, for Chapter 4's columns, the library and
# three definitions that leak on purpose.
# timeout: 300
import sqlite3

import pandas as pd

from foresight.checks.leakage import audit
from foresight.config import ML_WAREHOUSE
from foresight.features.build import load
from foresight.features.registry import Feature, names
from foresight.features.sources import load_sources
from foresight.segments import SEGMENT_DATE, account_features, fit_named
from foresight.train import COLUMNS


def to_the_end(ctx):
    """Tickets in the 90 days before the contract ends, read straight
    from the source instead of through window()."""
    e = ctx.keys.merge(ctx.sources.tickets, on="account_id")
    age = (e.end_date - e.day).dt.days
    e = e[(age >= 1) & (age <= 90)]
    return ctx.per_contract(e.groupby("contract_id").size())


def drifting_once(ctx):
    """Chapter 12's first behaviour column: Chapter 18's segmenter,
    fitted once, on 1 April 2024, then applied at every mark."""
    orders = ctx.sources.orders.reset_index(drop=True)
    accounts = ctx.sources.accounts.set_index("account_id")[["since"]]
    seg = fit_named(account_features(SEGMENT_DATE, orders, accounts))
    out = []
    for mark, k in ctx.keys.groupby("moment"):
        names_ = seg.label(account_features(mark, orders, accounts))
        out.append(k.account_id.map(names_).eq("Drifting")
                   .set_axis(k.contract_id))
    return pd.concat(out).reindex(ctx.index).astype(float)


def manager_today(ctx):
    """Never run: the audit fails it on its source alone."""
    raise NotImplementedError


LEAKS = (Feature("tickets_to_end", "leaks", "", ("tickets",), 90,
                 "raw", "tickets to the contract's end", to_the_end),
         Feature("drifting_once", "leaks", "", ("orders", "accounts"),
                 365, "raw", "segment from a fixed fit", drifting_once),
         Feature("manager_today", "leaks", "",
                 ("accounts.account_manager",), 0, "raw",
                 "today's account manager", manager_today))

table = load()
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
sources = load_sources()
columns = COLUMNS + names() + [f.name for f in LEAKS]
marks = sorted(train.moment.unique())
three = [marks[0], marks[len(marks) // 2], marks[-1]]
result = audit(train, columns, sources, marks=three, extra=LEAKS)
print(f"Deletion test on {train.moment.isin(three).sum():,} contracts,"
      " the cohorts marked\n" + ", ".join(f"{m:%Y-%m-%d}" for m in three)
      + "\n")
print(f"{'kind of source':<18}{'columns':>8}{'rows moved':>12}"
      f"{'fail':>6}")
for kind, g in result.groupby("kind", sort=False):
    moved = "-" if g.moved.isna().all() else f"{g.moved.sum():.0f}"
    print(f"{kind:<18}{len(g):>8}{moved:>12}"
          f"{(g.audit == 'fail').sum():>6}")
print("\nFailed")
for c, r in result[result.audit == "fail"].iterrows():
    moved = "not testable" if pd.isna(r.moved) else \
        f"{r.moved:.0f} rows moved"
    print(f"  {c:<16}{r.source:<26}{moved}")

# "Fixed at opening" is a claim; the old CRM's copy of each migrated
# account, frozen in February 2024, is one piece of evidence for it.
con = sqlite3.connect(ML_WAREHOUSE)
same, n = con.execute("""SELECT SUM(o.region_id = n.region_id),
    COUNT(*) FROM accounts o JOIN accounts n ON n.postcode = o.postcode
    AND n.since = o.since AND n.crm_source = 'Meridian CRM'
    WHERE o.crm_source = 'Legacy CRM'""").fetchone()
print(f"\nRegion in the old CRM and today: the same for {same} of {n}"
      " accounts")
