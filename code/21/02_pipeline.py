# v0.6 rebuilt as one scikit-learn object, checked against Chapter 16's
# model on the validation cohorts, then saved and loaded as one file.
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import calibrated
from foresight.evaluate import SPLITS, backtest, known_by, measure
from foresight.pipeline.model import INPUTS, build, maker
from foresight.train import make_model

table = pd.read_parquet(TABLE)
model = build(strength=0.002, months=3, no_order=365)

lasso = model.estimator
for name, step in lasso.steps:
    print(f"{name:<9}{type(step).__name__}")
for name, _, cols in lasso.named_steps["prepare"].transformers:
    print(f"  {name:<10}{', '.join(cols)}")

first, last = SPLITS["validation"]
old = backtest(table, first, last, calibrated(make_model()))
new = backtest(table, first, last, maker(model))
a, b = measure(old), measure(new)
calls = 240
print(f"\nValidation, {len(new):,} contracts{'Chapter 16':>15}"
      f"{'pipeline':>11}")
print(f"{'leavers in 240 calls':<30}"
      f"{a['precision']['model'] * calls:>10.0f}"
      f"{b['precision']['model'] * calls:>11.0f}")
print(f"{'AUC':<30}{a['auc']['model']:>10.4f}"
      f"{b['auc']['model']:>11.4f}")
gap = np.abs(old.model - new.model).max()
print(f"largest difference in any contract's chance: {gap:.1e}")

rows = known_by(table, "2024-10-02")
cohort = table[table.moment == "2024-10-02"]
fitted = build().fit(rows, rows.not_renewed)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model.joblib"
    joblib.dump(fitted, path)
    again = joblib.load(path)
    gap = np.abs(fitted.predict_proba(cohort)
                 - again.predict_proba(cohort)).max()
    size = path.stat().st_size / 1024
print(f"\nSaved as one file of {size:.0f} KB. Loaded again, its chances"
      f" for\nOctober's {len(cohort)} contracts differ by {gap:g}.")
names = fitted.estimator_.named_steps["prepare"].get_feature_names_out()
print(f"It reads {len(INPUTS)} columns of the table and makes"
      f" {len(names)} numbers.")
