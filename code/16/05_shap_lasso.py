# Shapley values for one contract and the lasso, three ways: by brute
# force over every group of columns, as weight times standardised
# value, and from the shap library. For a linear model they agree.
import numpy as np
import pandas as pd
import shap

from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.explain import COLUMNS, contributions
from foresight.train import make_model

table = pd.read_parquet(TABLE)
row = table[table.contract_id == 10222]          # Wickholm Foods
known = known_by(table[table.end_date >= HISTORY_FROM],
                 row.moment.iloc[0])
lasso = make_model()().fit(known)
w, b0 = lasso.model_.coef_[0], lasso.model_.intercept_[0]
x = lasso.scaler_.transform(lasso.columns(row).to_numpy(float))[0]
B = lasso.scaler_.transform(lasso.columns(known).to_numpy(float))
source = [c.split("=")[0].replace("log_", "") for c in lasso.columns_]
where = [np.array([s == c for s in source]) for c in COLUMNS]

# A group's value: the average log-odds when the group's columns take
# this contract's values and the rest take each training row's.
n = len(COLUMNS)
value = np.empty(2 ** n)
for mask in range(2 ** n):
    mixed = B.copy()
    for j in range(n):
        if mask >> j & 1:
            mixed[:, where[j]] = x[where[j]]
    value[mask] = (mixed @ w + b0).mean()

size = np.array([bin(m).count("1") for m in range(2 ** n)])
fact = np.array([np.prod(range(1, k + 1)) for k in range(n + 1)])
brute = np.zeros(n)
for j in range(n):
    for mask in range(2 ** n):
        if not mask >> j & 1:
            k = size[mask]
            share = fact[k] * fact[n - k - 1] / fact[n]
            brute[j] += share * (value[mask | 1 << j] - value[mask])

base, parts = contributions(lasso, row)
lib = shap.LinearExplainer(
    lasso.model_, shap.maskers.Independent(B, max_samples=len(B)))
sv = lib.shap_values(x[None, :])[0]
by_lib = [sv[where[j]].sum() for j in range(n)]

print(f"{'column':<17}{'value':>15}{'brute':>8}{'w times z':>10}"
      f"{'shap':>8}")
for j, c in enumerate(COLUMNS):
    v = row[c].iloc[0]
    v = f"{v:,.0f}" if isinstance(v, float) else str(v)
    print(f"{c:<17}{v:>15}{brute[j]:>8.3f}{parts[c].iloc[0]:>10.3f}"
          f"{by_lib[j]:>8.3f}")
print(f"{'base':<32}{value[0]:>8.3f}{base[0]:>10.3f}"
      f"{lib.expected_value:>8.3f}")
total = value[0] + brute.sum()
print(f"{'base + parts = log-odds':<32}{total:>8.3f}"
      f"{base[0] + parts.iloc[0].sum():>10.3f}")
p = lasso.predict_proba(row)[0]
print(f"log-odds of the chance, {p:.1%}: {np.log(p / (1 - p)):.3f}")
gap = np.abs(brute - parts[COLUMNS].iloc[0].to_numpy()).max()
print(f"Largest difference, brute force against w times z: {gap:.0e}")
