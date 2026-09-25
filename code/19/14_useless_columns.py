# Add columns of pure noise: which model is hurt more?
import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import (LIGHTGBM_DETERMINISTIC, rng,
                              seed_everything)
from foresight.data.build_table import TABLE
from foresight.evaluate import auc
from foresight.models.logistic import at_capacity, features
from foresight.models.mlp import RenewalMLP, known_split

seed_everything()
table = pd.read_parquet(TABLE)
train, valid = known_split(table)
noise = rng().normal(size=(table.contract_id.max() + 1, 100))


def with_noise(k):
    """Chapter 7's sixteen columns and k more of random numbers."""
    def columns(rows):
        extra = noise[rows.contract_id.to_numpy(), :k]
        names = [f"noise_{j}" for j in range(k)]
        return pd.concat([features(rows), pd.DataFrame(
            extra, columns=names, index=rows.index)], axis=1)
    return columns


print("Fixed split: validation AUC, and leavers in the top 40s")
print(f"{'useless columns':<17}{'LightGBM':>14}{'network':>14}"
      f"{'epochs':>8}")
for k in (0, 25, 50, 100):
    cols = with_noise(k)
    boost = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=200,
                           learning_rate=0.03, num_leaves=7,
                           min_child_samples=50)
    pb = boost.fit(cols(train), train.not_renewed).predict_proba(
        cols(valid))[:, 1]
    net = RenewalMLP(columns=cols).fit(train)
    pn = net.predict_proba(valid)
    y = valid.not_renewed
    hb, hn = (at_capacity(valid, p)["leavers"] for p in (pb, pn))
    print(f"{k:<17}{auc(y, pb):>8.3f}{hb:>6}{auc(y, pn):>8.3f}{hn:>6}"
          f"{net.epochs_:>8}")
