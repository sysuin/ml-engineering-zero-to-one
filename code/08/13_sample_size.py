# How many contracts would it take to tell two lists apart?
import numpy as np
import pandas as pd
from scipy.stats import norm

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, backtest, hits_at_k

table = pd.read_parquet(TABLE)
scored = backtest(table, *SPLITS["validation"])
cohorts = [(c.not_renewed.to_numpy(), c.model.to_numpy(),
            c.rule.to_numpy(), c.contract_id.to_numpy())
           for _, c in scored.groupby("moment")]


def paired_spread(months, reps=1000, seed=0):
    """Imagine `months` cohorts like these six: draw each from one of
    the six, contracts resampled. Return the spread (SD) of the model
    minus the rule, for AUC and for precision at 40."""
    g = rng(seed)
    d_auc, d_prec = [], []
    for _ in range(reps):
        y, m, r, hits = [], [], [], 0
        for i in g.integers(0, len(cohorts), months):
            cy, cm, cr, ids = cohorts[i]
            j = g.integers(0, len(cy), len(cy))
            y.append(cy[j])
            m.append(cm[j])
            r.append(cr[j])
            hits += (hits_at_k(cy[j], cm[j], ids[j])
                     - hits_at_k(cy[j], cr[j], ids[j]))
        y, m, r = (np.concatenate(v) for v in (y, m, r))
        d_auc.append(auc(y, m) - auc(y, r))
        d_prec.append(hits / (40 * months))
    return np.std(d_auc), np.std(d_prec), len(y)


def power(effect, spread):
    """Chance a 95% interval clears zero when the truth is `effect`."""
    return norm.cdf(effect / spread - 1.96)


print(f"{'':17}{'AUC gap: SD':>13}{'power':>8}"
      f"{'precision gap: SD':>19}{'power':>8}")
print(f"{'months':>6}{'contracts':>11}{'':>13}{'at 0.02':>8}"
      f"{'(points)':>19}{'at 3':>8}")
spreads = {}
for months in (6, 12, 24, 48):
    sa, sp, n = paired_spread(months)
    spreads[months] = (sa, sp, n)
    print(f"{months:>6}{n:>11,}{sa:>13.4f}{power(0.02, sa):>8.0%}"
          f"{sp * 100:>19.2f}{power(0.03, sp):>8.0%}")

# The spread falls as one over the square root of the size, so solve
# for the size that gives 80% power: effect = (1.96 + 0.84) x spread.
sa, sp, n = spreads[6]
z = norm.ppf(0.975) + norm.ppf(0.80)
for what, effect, s in (("0.02 of AUC", 0.02, sa),
                        ("3 points of precision", 0.03, sp)):
    months = 6 * (z * s / effect) ** 2
    print(f"80% power, {what}: about {months:.0f} months,"
          f" {months / 6 * n:,.0f} contracts")
