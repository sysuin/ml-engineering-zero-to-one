# One experiment, run in two fresh Python sessions on the same data
# with the same seed, gives two answers. Then the one-word fix.
# Python picks a new hash seed for every session unless told; the
# listing tells it, so that the page shows what two sessions would do.
import os
import subprocess
import sys

import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.data.build_table import TABLE
from foresight.models.logistic import log_loss

NOT_INPUTS = {"contract_id", "account_id", "moment", "end_date",
              "segment", "region", "not_renewed"}


def experiment(fixed: bool) -> str:
    """Fit on 2023's contracts, score April to June 2024."""
    t = pd.read_parquet(TABLE)
    fit = t[t.end_date.between("2023-01-01", "2023-12-31")]
    watch = t[t.end_date.between("2024-04-01", "2024-06-30")]
    inputs = set(t.columns) - NOT_INPUTS        # every other column
    inputs = sorted(inputs) if fixed else list(inputs)
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=300,
                       learning_rate=0.05, num_leaves=4,
                       colsample_bytree=0.5)
    m.fit(fit[inputs].astype(float), fit.not_renewed)
    p = m.predict_proba(watch[inputs].astype(float))[:, 1]
    loss = log_loss(watch.not_renewed.to_numpy(), p)
    return f"{loss:.5f}  {', '.join(inputs[:2])}, ..."


if len(sys.argv) > 1:                   # inside one session
    print(experiment(sys.argv[1] == "fixed"))
    sys.exit()

for version in ("as written", "fixed"):
    print(f"{version}:")
    for session in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": session}
        out = subprocess.run([sys.executable, __file__, version],
                             env=env, capture_output=True, text=True)
        print(f"  session {session}  log loss {out.stdout.strip()}")
