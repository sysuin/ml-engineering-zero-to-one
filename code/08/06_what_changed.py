# Did small businesses and discounts matter differently after July 2023?
import numpy as np
from sklearn.linear_model import LogisticRegression

from foresight.config import rng
from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, VALIDATION, features, load

CHANGE = "2023-07-01"           # small-business prices rose this day


def with_change(rows):
    """Chapter 7's columns, plus 'after the change' and two columns
    that let the small-business and discount weights differ after."""
    X = features(rows)
    after = (rows.moment >= CHANGE).astype(float)
    small = (rows.segment.fillna("Small business")
             == "Small business").astype(float)
    X["after"] = after
    X["small business, after"] = small * after
    X["discount, after"] = X.discount_pct * after
    return X


def weights(X, y):
    """Unpenalised logistic weights, per unit of each column."""
    scale = Standardiser().fit(X)
    fit = LogisticRegression(C=np.inf, max_iter=1000)
    fit.fit(scale.transform(X), y)
    return fit.coef_[0] / scale.scale_


SHOW = ["small business, after", "discount, after"]
for label, last in (("training split", TRAIN[1]),
                    ("training and validation", VALIDATION[1])):
    rows = load(TRAIN[0], last)
    X = with_change(rows)
    cols = [X.columns.get_loc(c) for c in SHOW]
    X, y = X.to_numpy(), rows.not_renewed.to_numpy()
    w = weights(X, y)[cols]
    # Resample accounts, not rows: an account's contracts go together.
    g, by_account = rng(), rows.groupby("account_id").indices
    accounts = list(by_account)
    draws = []
    for _ in range(300):
        pick = g.choice(len(accounts), len(accounts))
        ix = np.concatenate([by_account[accounts[i]] for i in pick])
        draws.append(weights(X[ix], y[ix])[cols])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    print(f"{label}: {len(rows):,} contracts,"
          f" {(rows.moment >= CHANGE).sum():,} after the change")
    for name, v, a, b in zip(SHOW, w, lo, hi):
        print(f"  {name:<22}{v:>+7.3f}   95% interval {a:+.3f}"
              f" to {b:+.3f}")
    print(f"  small-business odds after, times {np.exp(w[0]):.2f}")
