# The showdown: network, boosting, logistic regression and the rule.
# timeout: 300
import json

import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC, seed_everything
from foresight.data.build_table import TABLE
from foresight.evaluate import (SPLITS, backtest, bootstrap, interval,
                                measure)
from foresight.models.logistic import RenewalRisk, features
from foresight.models.mlp import RenewalMLP

seed_everything()                     # after torch, which mlp imports


class Boosted:
    """LightGBM on the same sixteen columns, with modest settings."""

    def fit(self, rows):
        self.model_ = LGBMClassifier(
            **LIGHTGBM_DETERMINISTIC, n_estimators=200,
            learning_rate=0.03, num_leaves=7, min_child_samples=50)
        self.model_.fit(features(rows), rows.not_renewed)
        return self

    def predict_proba(self, rows):
        return self.model_.predict_proba(features(rows))[:, 1]


table = pd.read_parquet(TABLE)
models = {"logistic (Ch. 7)": RenewalRisk, "LightGBM": Boosted,
          "network": RenewalMLP}
point, draws = {}, {}
for name, make in models.items():
    scored = backtest(table, *SPLITS["validation"], make)
    m, d = measure(scored), bootstrap(scored)   # same seed, same draws
    point[name] = {k: m[k]["model"] for k in ("precision", "auc")}
    draws[name] = {k: d[(k, "model")] for k in ("precision", "auc")}
point["the rule"] = {k: m[k]["rule"] for k in ("precision", "auc")}
draws["the rule"] = {k: d[(k, "rule")] for k in ("precision", "auc")}

calls = int(scored.groupby("moment").size().clip(upper=40).sum())
print(f"Validation backtest: {scored.moment.nunique()} cohorts,"
      f" {int(scored.not_renewed.sum())} leavers, {calls} calls")
print(f"{'':16}{'leavers':>8}{'precision % (95%)':>20}"
      f"{'AUC (95%)':>20}")
for name in ("the rule", *models):
    p, a = point[name]["precision"], point[name]["auc"]
    plo, phi = interval(draws[name]["precision"])
    alo, ahi = interval(draws[name]["auc"])
    print(f"{name:<16}{p * calls:>8.0f}{p * 100:>7.1f} ({plo * 100:.1f}"
          f"-{phi * 100:.1f}){a:>9.3f} ({alo:.3f}-{ahi:.3f})")
print(f"For scale: a random list {m['precision']['random']:.1%},"
      f" a perfect one {m['precision']['perfect']:.1%}")

print("\nPaired differences in precision, points (95%)")
pairs = [("network", "LightGBM"), ("network", "logistic (Ch. 7)"),
         ("LightGBM", "logistic (Ch. 7)"), ("network", "the rule")]
for a, b in pairs:
    diff = (draws[a]["precision"] - draws[b]["precision"]) * 100
    lo, hi = interval(diff)
    gap = (point[a]["precision"] - point[b]["precision"]) * 100
    print(f"  {a + ' - ' + b:<30}{gap:>+6.1f} ({lo:+.1f} to {hi:+.1f})")

with open("code/19/12_tabular_showdown.json", "w") as f:
    json.dump({name: {k: [point[name][k], *interval(draws[name][k])]
                      for k in ("precision", "auc")}
               for name in point}
              | {"calls": calls, "random": m["precision"]["random"],
                 "perfect": m["precision"]["perfect"]}, f)
