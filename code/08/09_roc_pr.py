# ROC and precision-recall curves, built by walking down the list.
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, backtest

table = pd.read_parquet(TABLE)
scored = backtest(table, *SPLITS["validation"])
y = scored.not_renewed.to_numpy()


def walk(y, score):
    """Lower the threshold one distinct score at a time, and count
    the leavers and renewers above it after each step."""
    order = np.argsort(-score, kind="stable")
    s, t = score[order], y[order]
    step = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # tie runs end
    tp, fp = np.cumsum(t)[step], np.cumsum(1 - t)[step]
    recall = tp / t.sum()                  # also the true positive rate
    fpr = fp / (len(t) - t.sum())          # the false positive rate
    return np.r_[0, fpr], np.r_[0, recall], tp / (tp + fp)


print(f"Validation: {len(y):,} contracts, {y.sum()} leavers "
      f"({y.mean():.1%})")
print(f"{'':7}{'area under ROC':>16}{'library':>9}{'rank AUC':>10}"
      f"{'avg precision':>15}{'library':>9}")
curves = {}
for name in ("model", "rule"):
    s = scored[name].to_numpy(dtype=float)
    fpr, rec, prec = walk(y, s)
    area = np.trapezoid(rec, fpr)
    ap = np.sum(np.diff(rec) * prec)     # precision at each new leaver
    print(f"{name:7}{area:>16.4f}{roc_auc_score(y, s):>9.4f}"
          f"{auc(y, s):>10.4f}{ap:>15.4f}"
          f"{average_precision_score(y, s):>9.4f}")
    keep = np.unique(np.linspace(0, len(prec) - 1, 200).astype(int))
    curves[name] = {"fpr": fpr[keep + 1].round(4).tolist(),
                    "recall": rec[keep + 1].round(4).tolist(),
                    "precision": prec[keep].round(4).tolist()}
print(f"{'random':7}{0.5:>16.4f}{'':>19}{y.mean():>15.4f}")

# Where the account team works: its 240 calls, top 40 a cohort.
top = scored.sort_values(["moment", "model", "contract_id"],
                         ascending=[True, False, True])
top = top.groupby("moment").head(40)
tp = int(top.not_renewed.sum())
print(f"\nThe model's 240 calls: {tp} leavers, {240 - tp} renewers")
print(f"  true positive rate (recall) {tp / y.sum():.1%}")
print(f"  false positive rate {(240 - tp) / (len(y) - y.sum()):.1%}"
      " of renewers")
with open("code/08/09_roc_pr.json", "w") as f:
    json.dump({"curves": curves, "base": float(y.mean()),
               "at": [(240 - tp) / (len(y) - y.sum()), tp / y.sum(),
                      tp / 240]}, f)
