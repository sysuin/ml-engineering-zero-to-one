# What the monthly job left behind: the tables, the list as a
# dashboard reads it beside the warehouse, and the rows it scored,
# which give the same chances when they are scored again.
import json
from contextlib import closing

import pandas as pd

from foresight.pipeline.registry import Registry
from foresight.serve import SANDBOX, online, store

scores = store.path(SANDBOX)
print("scores.db")
for t in ("runs", "scores", "inputs", "forecasts"):
    n = store.read(scores, f"SELECT COUNT(*) AS n FROM {t}").n[0]
    print(f"  {t:<10}{n:>5} rows")

DASHBOARD = """
    SELECT a.rank, a.contract_id, c.name, a.chance, a.arm
    FROM ml.answers a JOIN accounts c USING (account_id)
    WHERE a.listed = 1 ORDER BY a.rank"""
with closing(online.open_warehouse()) as con:
    con.execute(f"ATTACH DATABASE '{scores.resolve().as_uri()}"
                "?mode=ro' AS ml")
    board = pd.read_sql_query(DASHBOARD, con)
board["chance"] = board.chance.map("{:.1%}".format)
print("\nThe dashboard's query: the warehouse, the scores beside it")
print(board.head(6).to_string(index=False))
print(f"  ... {len(board)} listed:"
      f" {(board.arm == 'called').sum()} to call,"
      f" {(board.arm == 'held out').sum()} held out")

registry = Registry("renewal", SANDBOX / "artifacts")
model, manifest = registry.production()
kept = store.inputs(scores, 1, manifest)
stored = store.read(scores, "SELECT contract_id, chance FROM scores"
                    " WHERE run = 1 AND key_account = 0"
                    " ORDER BY contract_id")
again = model.predict_proba(kept[kept.contract_id.isin(
    stored.contract_id)])[:, 1]
print(f"\nThe {len(kept)} rows run 1 scored, scored again: largest"
      f" change {abs(again - stored.chance).max():.1e}")

one = store.read(scores, "SELECT * FROM scores WHERE run = 1"
                 " AND rank = 1").iloc[0]
run = store.read(scores, "SELECT run, job, mark, run_on, source,"
                 " model_as_of, status, step, rows FROM runs"
                 " WHERE run = 1").iloc[0]
with open("code/22/05_write_back.json", "w") as f:
    json.dump({"score": one.to_dict(), "run": run.to_dict()}, f,
              default=str)
