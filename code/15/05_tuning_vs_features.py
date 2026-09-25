# Tuning gains against a feature's gain on one scale: each model's log
# loss on its tuning cohorts, from its first settings to its searched
# ones, with Chapter 12's order_trend, and the true probabilities.
# timeout: 300
import json
from pathlib import Path

import pandas as pd

from foresight.config import TRUTH
from foresight.data.build_table import TABLE
from foresight.features.build import load
from foresight.models.featured import (TRAINING, booster, lasso,
                                       tune_lasso, tuning_score,
                                       watched_loss)
from foresight.models.logistic import log_loss
from foresight.models.regularised import TUNE
from foresight.evaluate import backtest
from foresight.tune import TunableBooster, best, score, search

table = load()                      # Chapter 4's rows + the library
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")


def floor(scored):
    """The log loss of the true probabilities on the same rows."""
    p = truth.p_leave.reindex(scored.contract_id).to_numpy()
    return log_loss(scored.not_renewed.to_numpy(), p)


chosen = best(search(pd.read_parquet(TABLE)))  # listing 11's search
steps = {"booster": [
    ("31 leaves of 20", score(table, {"leaves": 31, "min_leaf": 20})),
    ("Chapter 11's stumps", score(table)),
    ("the search's best", score(table, chosen)),
    ("stumps + order_trend",
     score(table, make_model=booster(["order_trend"])))]}
steps["booster"].append(("true probabilities", {
    "log loss": floor(steps["booster"][1][1]["scored"])}))

fine = tune_lasso(table, (), [0.0005, 0.001, 0.0015, 0.002, 0.003,
                              0.004, 0.006, 0.01])
strength = float(fine.loc[fine["log loss"].idxmin(), "strength"])
steps["lasso"] = [
    ("no penalty", tuning_score(table, lasso((), 0.0))),
    ("Chapter 9's penalty", tuning_score(table, lasso())),
    (f"finer search: {strength:g}", tuning_score(table,
                                                 lasso((), strength))),
    ("+ order_trend", tuning_score(table, lasso(["order_trend"]))),
    ("true probabilities",
     {"log loss": floor(backtest(table, *TUNE, lasso()))})]

for model, rows in steps.items():
    cohorts = "Oct 2023 to Apr 2024" if model == "booster" else \
        "Jul 2023 to Apr 2024"
    print(f"{model}, cohorts ending {cohorts}")
    print(f"  {'':26}{'log loss':>9}{'fall':>9}{'AUC':>7}"
          f"{'leavers':>8}")
    loss = [s["log loss"] for _, s in rows]
    # Each step's fall from the row it starts at: the first settings,
    # then the tuned or featured model, then the better of those two.
    start = [None, loss[0], loss[1], loss[1], min(loss[2], loss[3])]
    for (name, s), was in zip(rows, start):
        fall = f"{was - s['log loss']:>9.5f}" if was else f"{'':9}"
        rest = (f"{s['auc']:>7.3f}{s.get('leavers', s.get('hits')):>8}"
                if "auc" in s else "")
        print(f"  {name:<26}{s['log loss']:>9.5f}{fall}{rest}")

train = table[table.end_date.between(*TRAINING)]
print("\nThe booster on Chapter 11's watched months (Chapter 12)")
for name, loss in (
        ("31 leaves of 20", watched_loss(table, (), 31, 20)[0]),
        ("Chapter 11's stumps", watched_loss(table)[0]),
        ("the search's best",
         min(TunableBooster(**chosen).fit(train).watched_)),
        ("stumps + order_trend",
         watched_loss(table, ["order_trend"])[0])):
    print(f"  {name:<26}{loss:>9.5f}")

with open(Path(__file__).with_suffix(".json"), "w") as f:
    json.dump({m: [[n, s["log loss"]] for n, s in rows]
               for m, rows in steps.items()}, f)
