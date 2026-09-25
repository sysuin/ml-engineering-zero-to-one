# Equal miss rates, at a price: reserve some of each cohort's 40 calls
# for Mid-market contracts, the rest by chance, and count what the list
# finds, segment by segment, on the validation cohorts.
import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, calibrated, net_value
from foresight.evaluate import SPLITS, backtest
from foresight.models.logistic import CALLS
from foresight.train import make_model

rows = pd.read_parquet(TABLE)
s = backtest(rows, *SPLITS["validation"], calibrated(make_model()))
mid = (s.segment == "Mid-market").to_numpy()
small = (s.segment == "Small business").to_numpy()
y = s.not_renewed.to_numpy()


def reserved(m):
    """The top m Mid-market contracts of each cohort, then the rest of
    the 40 by chance."""
    first = by_capacity(s[mid], k=m) if m else np.zeros(mid.sum(), bool)
    called = np.zeros(len(s), bool)
    called[np.flatnonzero(mid)[first]] = True
    rest = s.assign(model=np.where(called, np.inf, s.model))
    return by_capacity(rest, k=CALLS)


print(f"{'reserved':>8}{'Mid calls':>11}{'Mid missed':>12}"
      f"{'Small missed':>14}{'leavers':>9}{'made $':>9}")
for m in (0, 4, 8, 12, 16, 20):
    c = reserved(m)
    miss = [1 - c[g & (y == 1)].mean() for g in (mid, small)]
    print(f"{m:>8}{c[mid].sum():>11}{miss[0]:>12.0%}{miss[1]:>14.0%}"
          f"{y[c].sum():>9}{net_value(y, c):>+9,.0f}")
print(f"Mid-market leavers: {y[mid].sum()} of {mid.sum()} contracts;"
      f" Small business: {y[small].sum()} of {small.sum()}")
