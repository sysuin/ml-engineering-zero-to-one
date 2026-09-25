# What tuning could be worth: the booster with big trees, as Chapter 11
# set it, beside the lasso and the best any model could do, all on the
# training period's tuning cohorts. The truth file gives the best only.
import pandas as pd

from foresight.config import TRUTH
from foresight.data.build_table import TABLE
from foresight.evaluate import auc, hits_at_k
from foresight.models.logistic import log_loss
from foresight.models.regularised import maker
from foresight.tune import TUNING, score

table = pd.read_parquet(TABLE)
lists = {"booster, 31 leaves of 20": score(table, {"leaves": 31,
                                                   "min_leaf": 20}),
         "booster, Chapter 11": score(table),
         "lasso, v0.4": score(table, make_model=maker("l1", 0.002))}

s = lists["booster, Chapter 11"]["scored"]
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
best = s.assign(model=truth.p_leave.reindex(s.contract_id).to_numpy())
y, p = best.not_renewed.to_numpy(), best.model.to_numpy()
lists["the true probabilities"] = {
    "log loss": log_loss(y, p), "auc": auc(y, p),
    "leavers": sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
                   for _, c in best.groupby("moment"))}

calls = int(s.groupby("moment").size().clip(upper=40).sum())
print(f"Tuning cohorts: contracts ending {TUNING[0]} to {TUNING[1]}")
print(f"{s.moment.nunique()} cohorts, {len(s):,} contracts,"
      f" {int(s.not_renewed.sum())} leavers, {calls} calls\n")
print(f"{'':28}{'log loss':>9}{'AUC':>7}{'leavers':>9}")
for name, r in lists.items():
    print(f"{name:<28}{r['log loss']:>9.5f}{r['auc']:>7.3f}"
          f"{r['leavers']:>9}")

ours = lists["booster, Chapter 11"]
top = lists["the true probabilities"]
print("\nChapter 11's booster to the true probabilities:")
print(f"  log loss {ours['log loss'] - top['log loss']:.5f},"
      f" AUC {top['auc'] - ours['auc']:.3f},"
      f" leavers {top['leavers'] - ours['leavers']}")
