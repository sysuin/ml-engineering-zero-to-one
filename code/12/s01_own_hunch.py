# Exercise 1: a hunch of one's own, "accounts that move their ordering
# to the web have stopped talking to us", as a library-style feature.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import legacy_ids
from foresight.evaluate import SPLITS, backtest
from foresight.features.build import load
from foresight.features.sources import Context, load_sources
from foresight.models.featured import (SHOWN, TRAINING, gain, lasso,
                                       shown, tuning_score)

src = load_sources()
with sqlite3.connect(ML_WAREHOUSE) as con:
    web = pd.read_sql_query("""
        SELECT account_id, order_date AS day, channel = 'Web' AS value
        FROM orders""", con)
    web["account_id"] = web.account_id.replace(legacy_ids(con))
web["day"] = pd.to_datetime(web.day)
src.web = web                           # one more source for window()

table = load()
ctx = Context(table, src)
e = ctx.window("web", 90)
table["web_share_90d"] = ctx.per_contract(
    e.groupby("contract_id").value.mean()).to_numpy()

train = table[table.end_date.between(*TRAINING)]
band = pd.cut(train.web_share_90d, [-0.01, 0, 0.5, 0.99, 1],
              labels=["none", "under half", "half or more", "all"])
r = train.groupby(band, observed=True).not_renewed.agg(["size", "mean"])
print(f"{'Training, web share':<24}{'contracts':>10}{'left':>8}")
for b, row in r.iterrows():
    print(f"  {b:<22}{row['size']:>10,.0f}{row['mean']:>8.1%}")

print(f"\n{'Tuning cohorts':<28}{'log loss':>9}{'AUC':>7}")
for name, extra in (("v0.4's lasso", ()),
                    ("  + web_share_90d", ["web_share_90d"])):
    s = tuning_score(table, lasso(extra))
    print(f"  {name:<26}{s['log loss']:>9.5f}{s['auc']:>7.3f}")

V = SPLITS["validation"]
g = gain(backtest(table, *V, lasso(["web_share_90d"])),
         backtest(table, *V, lasso()))
print(f"\nValidation, with minus without\n  {'':10}{SHOWN}")
print(f"  {'lasso':<13}{shown(g)}")
