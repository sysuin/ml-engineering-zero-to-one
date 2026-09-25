# The forest and the logistic model, each without days since the last
# order, against their full versions on the validation backtest.
# timeout: 300
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from foresight.config import SEED
from foresight.leaderboard import paired, point, run
from foresight.models.forest import (COLUMNS, MIN_LEAF, TREES,
                                     RenewalForest)
from foresight.models.linear import Standardiser
from foresight.models.logistic import RenewalRisk, features


def columns(rows):
    return features(rows).drop(columns="days_since_order").to_numpy()


class ForestNoGap:
    def fit(self, rows):
        self.m = RandomForestClassifier(
            n_estimators=TREES, min_samples_leaf=MIN_LEAF,
            max_features=COLUMNS, random_state=SEED, n_jobs=1)
        self.m.fit(columns(rows), rows.not_renewed)
        return self

    def predict_proba(self, rows):
        return self.m.predict_proba(columns(rows))[:, 1]


class LogisticNoGap:
    def fit(self, rows):
        self.s = Standardiser().fit(columns(rows))
        self.m = LogisticRegression(C=np.inf, tol=1e-10,
                                    max_iter=10_000)
        self.m.fit(self.s.transform(columns(rows)), rows.not_renewed)
        return self

    def predict_proba(self, rows):
        Z = self.s.transform(columns(rows))
        return self.m.predict_proba(Z)[:, 1]


board = run({"logistic": RenewalRisk, "logistic, no gap": LogisticNoGap,
             "forest": RenewalForest, "forest, no gap": ForestNoGap})
print(f"{'':18}{'leavers':>8}{'AUC':>7}")
for name in board["runs"]:
    print(f"{name:<18}{point(board, 'precision', name) * 240:>8.0f}"
          f"{point(board, 'auc', name):>7.3f}")
print("\nFull minus no gap, paired (95%)")
for m in ("logistic", "forest"):
    p, plo, phi = paired(board, "precision", m, f"{m}, no gap")
    a, alo, ahi = paired(board, "auc", m, f"{m}, no gap")
    print(f"  {m:<9} precision {p * 100:+.1f} ({plo * 100:+.1f} to"
          f" {phi * 100:+.1f})")
    print(f"  {'':<9} AUC {a:+.3f} ({alo:+.3f} to {ahi:+.3f})")
