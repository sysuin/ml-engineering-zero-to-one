# The same runs in MLflow: each with its settings, scores, data and code
# hashes and the fitted model, in a local store, then compared.
# timeout: 300
import json
import tempfile
from pathlib import Path

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.models.regularised import RegularisedRisk
from foresight.tracking import open_store, runs, track
from foresight.tune import (CHAPTER_11, TRAINING, TunableBooster, best,
                            score, search)

table = pd.read_parquet(TABLE)
rows = table[table.end_date.between(*TRAINING)]
store = tempfile.TemporaryDirectory()   # a fresh store for the page


def log(name, model, params, make=None):
    s = score(table, params if make is None else None, make_model=make)
    metrics = {k: s[k] for k in ("log loss", "auc", "leavers")}
    track(name, model.fit(rows), params, rows, table, metrics,
          store=store.name, log=f"{store.name}/runs.jsonl")


lasso = {"penalty": "l1", "strength": 0.002}
log("lasso, v0.4", RegularisedRisk(**lasso), lasso,
    lambda: RegularisedRisk(**lasso))
log("booster, Chapter 11", TunableBooster(), CHAPTER_11)
tuned = best(search(table))
log("booster, tuned", TunableBooster(**tuned), tuned)

found = runs(store.name)
print(f"{len(found)} runs in experiment 'foresight-renewals'\n")
view = found[["tags.mlflow.runName", "params.leaves", "params.rate",
              "params.strength", "metrics.log_loss", "metrics.auc",
              "metrics.leavers"]].copy()
view.columns = ["run name", "leaves", "rate", "strength", "log loss",
                "AUC", "leavers"]
view["rate"] = view.rate.astype(float).round(3)
view["leavers"] = view.leavers.astype(int)
print(view.round({"log loss": 5, "AUC": 3}).fillna("")
      .to_string(index=False))

mlflow = open_store(store.name)
top = mlflow.search_runs(experiment_names=["foresight-renewals"],
                         order_by=["metrics.log_loss ASC"],
                         max_results=1).iloc[0]
print(f"\nLowest log loss: {top['tags.mlflow.runName']}")
print(f"  data {top['tags.data.content_sha256'][:16]}...,"
      f" rows {top['tags.data.rows']}")
print(f"  code {top['tags.code.sha256'][:16]}..., from")
for name in top["tags.code.files"].split(","):
    print(f"    {name}")
artifacts = mlflow.MlflowClient().list_artifacts(top.run_id)
print(f"  artifacts: {', '.join(a.path for a in artifacts)}")

with open(Path(__file__).with_suffix(".json"), "w") as f:
    data = found["tags.data.content_sha256"].str[:8].tolist()
    code = found["tags.code.sha256"].str[:8].tolist()
    json.dump({"runs": view.to_dict("records"), "data": data,
               "code": code}, f)
store.cleanup()
