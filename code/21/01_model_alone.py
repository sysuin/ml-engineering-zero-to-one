# The notebook's way: the model saved on its own, its preprocessing
# left behind in the cells, and a scoring script written later that has
# to redo it. The six validation cohorts, scored both ways.
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, SPLITS, hits_at_k, known_by
from foresight.models.logistic import features

table = pd.read_parquet(TABLE)
history = table[table.end_date >= HISTORY_FROM]
cohorts = table[table.end_date.between(*SPLITS["validation"])]


def notebook(rows):
    """Cells 4 to 9: numbers, a scaler, Chapter 9's lasso. The scaler
    stays in the notebook's memory; only the model is pickled."""
    X = features(rows).to_numpy()
    scaler = StandardScaler().fit(X)
    lasso = LogisticRegression(
        l1_ratio=1, solver="liblinear", C=1 / (0.002 * len(rows)),
        tol=1e-8, intercept_scaling=100, max_iter=10_000,
        random_state=SEED).fit(scaler.transform(X), rows.not_renewed)
    return pickle.dumps(lasso), scaler


def scoring_script(blob, cohort):
    """Written months later from the pickle alone. It standardises,
    as the notebook did, with the only rows it has: the cohort."""
    lasso = pickle.loads(blob)
    Z = StandardScaler().fit_transform(features(cohort).to_numpy())
    return lasso.predict_proba(Z)[:, 1]


print(f"{'mark':<11}{'rows':>5}{'mean chance':>18}{'largest':>9}"
      f"{'calls':>7}{'leavers':>10}")
print(f"{'':<16}{'notebook':>10}{'script':>8}{'change':>9}"
      f"{'moved':>7}{'nb':>5}{'sc':>5}")
total = np.zeros(3, int)
for mark, cohort in cohorts.groupby("moment"):
    blob, scaler = notebook(known_by(history, mark))
    Z = scaler.transform(features(cohort).to_numpy())
    right = pickle.loads(blob).predict_proba(Z)[:, 1]
    later = scoring_script(blob, cohort)
    ids, y = cohort.contract_id.to_numpy(), cohort.not_renewed
    top = [set(ids[np.lexsort((ids, -p))[:40]]) for p in (right, later)]
    hits = [hits_at_k(y, p, ids) for p in (right, later)]
    total += [len(top[0] - top[1]), *hits]
    print(f"{mark:%Y-%m-%d} {len(cohort):>5}{right.mean():>10.1%}"
          f"{later.mean():>8.1%}{np.abs(right - later).max():>9.3f}"
          f"{len(top[0] - top[1]):>7}{hits[0]:>5}{hits[1]:>5}")
print(f"{'six cohorts':<16}{'':>18}{'':>9}{total[0]:>7}{total[1]:>5}"
      f"{total[2]:>5}")
with open("code/21/01_model_alone.json", "w") as f:
    json.dump({"moved": int(total[0]), "notebook": int(total[1]),
               "script": int(total[2])}, f)
