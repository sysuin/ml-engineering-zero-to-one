# SHAP values for the booster on the October cohort: the shap library's
# TreeExplainer against LightGBM's own, the additivity check, and the
# average size of each column's part for the booster and the lasso.
import warnings

import numpy as np
import pandas as pd
import shap

from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.explain import COLUMNS, contributions, log_odds
from foresight.models.boosting import RenewalBooster, columns
from foresight.train import make_model

# shap announces a change to its output format on every call.
warnings.filterwarnings("ignore", message="LightGBM binary classifier")

table = pd.read_parquet(TABLE)
mark = pd.Timestamp("2024-10-02")
cohort = table[table.moment == mark]
known = known_by(table[table.end_date >= HISTORY_FROM], mark)
booster = RenewalBooster().fit(known)
lasso = make_model()().fit(known)

tree = shap.TreeExplainer(booster.model_)
by_shap = tree.shap_values(columns(cohort))
base, parts = contributions(booster, cohort)      # LightGBM's own
print(f"{len(cohort)} contracts, {booster.rounds_} stumps")
print(f"TreeExplainer against LightGBM, largest difference:"
      f" {np.abs(by_shap - parts.to_numpy()).max():.0e}")
total = base + parts.sum(axis=1).to_numpy()
p = booster.predict_proba(cohort)
print(f"base + parts against the log-odds, largest difference:"
      f" {np.abs(total - log_odds(p)).max():.0e}")
chance = 1 / (1 + np.exp(-base[0]))
print(f"base: {base[0]:.3f}, the log-odds of {chance:.1%}")

lb, lparts = contributions(lasso, cohort)
size = pd.DataFrame({"booster": parts[COLUMNS].abs().mean(),
                     "lasso": lparts[COLUMNS].abs().mean()})
size = size.sort_values("booster", ascending=False)
print(f"\nAverage size of each column's part"
      f"{'booster':>18}{'lasso':>8}")
for c, r in size.iterrows():
    print(f"  {c:<45}{r.booster:>6.3f}{r.lasso:>8.3f}")
