# What Chapter 16's score.py makes on the last morning of 2025, and
# whether the serving path gives its lists when both can make one.
import pandas as pd

from foresight import score as old
from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.pipeline import features
from foresight.pipeline.artifact import schema
from foresight.pipeline.model import build
from foresight.serve import renewal
from foresight.train import COLUMNS

tonight = features.mark_on("2025-12-31")
due = features.at_mark(tonight)
made = old.score("2025-12-31", checks=False)
print(f"Due on 2025-12-31: the cohort marked {tonight:%Y-%m-%d},"
      f" {len(due)} contracts")
print(f"score.py makes: the cohort marked {made['mark']:%Y-%m-%d},"
      f" {made['contracts']} contracts")
table = pd.read_parquet(TABLE)
print(f"  the latest mark in its table: {table.moment.max():%Y-%m-%d}")

# A morning both can serve: the list marked 2024-12-31.
before = old.score("2025-01-01", checks=False)
rows = known_by(table[table.end_date >= HISTORY_FROM], "2025-01-01")
model = build().fit(rows, rows.not_renewed)
cohort = features.at_mark(before["mark"])
new = renewal.score(model, {"inputs": schema(rows, COLUMNS)}, cohort,
                    rows)
a = before["list"].set_index("contract_id")
b = new[new.listed].set_index("contract_id").reindex(a.index)
print(f"\nThe list marked {before['mark']:%Y-%m-%d}, made both ways")
print(f"  contracts on both lists     {b.chance.notna().sum()} of"
      f" {len(a)}")
print(f"  same rank                   {(a['rank'] == b['rank']).sum()}")
print(f"  same reasons                "
      f"{(a.reasons == b.reasons).sum()}")
print(f"  largest gap in a chance     "
      f"{(a.chance - b.chance).abs().max():.1e}")
