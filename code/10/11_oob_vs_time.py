# Out-of-bag against the backtest, on the same contracts as Chapter 8.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, known_by, measure
from foresight.models.logistic import features


def forest(oob=False):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                  oob_score=oob, random_state=SEED,
                                  n_jobs=1)


table = pd.read_parquet(TABLE)
pool = table[table.end_date <= SPLITS["validation"][1]]
pool = pool.reset_index(drop=True)      # 2023 and 2024, as in §8.5
later = (pool.moment >= "2023-07-01").to_numpy()
X, y = features(pool), pool.not_renewed

# Out-of-bag: one forest on all 7,124; each row scored by the trees
# whose bootstrap sample missed it, trees that saw rows from after it.
bag = forest(oob=True).fit(X, y).oob_decision_function_[:, 1]

# The backtest: each cohort scored by a forest fitted only on the
# outcomes known at its mark.
known = np.full(len(pool), np.nan)
for mark in pool.moment[later].unique():
    now = (pool.moment == mark).to_numpy()
    past = known_by(pool, mark)
    known[now] = forest().fit(features(past), past.not_renewed) \
        .predict_proba(X[now])[:, 1]

rows = pool[later]
print(f"The same {len(rows):,} contracts, {rows.moment.nunique()}"
      f" cohorts, {rows.not_renewed.sum()} leavers")
print(f"  {'scored by':<25}{'AUC':>7}{'top-40 leavers':>16}")
for name, p in (("out-of-bag", bag),
                ("backtest, known by mark", known)):
    m = measure(rows.assign(model=p[later], rule=0.0, base=0.1))
    hits = round(m["precision"]["model"] * 40 * rows.moment.nunique())
    print(f"  {name:<25}{m['auc']['model']:>7.3f}{hits:>16}")
