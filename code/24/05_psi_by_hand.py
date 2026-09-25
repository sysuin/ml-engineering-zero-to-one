# The population stability index by hand, for Voss's share of spend:
# bins cut at the deciles of the rows v0.6 learned from, then the
# March 2025 list counted into the same bins.
import numpy as np

from foresight.evaluate import HISTORY_FROM
from foresight.monitor.cohorts import GO_LIVE, as_scored, known
from foresight.monitor.psi import Baseline, psi, shares

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
march = rows[rows.moment == "2025-03-02"]
x_old, x_new = train.supplier_voss, march.supplier_voss

cuts = np.unique(np.quantile(x_old, np.linspace(0.1, 0.9, 9)))
old = np.bincount(np.searchsorted(cuts, x_old), minlength=len(cuts) + 1)
new = np.bincount(np.searchsorted(cuts, x_new), minlength=len(cuts) + 1)
a, b = old / old.sum(), new / new.sum()
print(f"{'bin, Voss share':<18}{'training':>10}{'March':>8}"
      f"{'new - old':>11}{'ln(new/old)':>13}{'term':>7}")
lows = np.r_[0, cuts]
for i in range(len(a)):
    top = f"{cuts[i]:.3f}" if i < len(cuts) else "1"
    span = f"{lows[i]:.3f} to {top}" if i else "exactly 0"
    t = (b[i] - a[i]) * np.log(b[i] / a[i])
    print(f"{span:<18}{a[i]:>10.3f}{b[i]:>8.3f}{b[i] - a[i]:>+11.3f}"
          f"{np.log(b[i] / a[i]):>+13.3f}{t:>7.3f}")
total = ((b - a) * np.log(b / a)).sum()
print(f"{'PSI, by hand':<60}{total:>7.3f}")
print(f"{'PSI, foresight.monitor.psi':<60}"
      f"{psi(shares(x_old, cuts), shares(x_new, cuts)):>7.3f}")

print("\nThe same column against last month instead of training")
base = Baseline(train, ["supplier_voss"])
marks = sorted(rows[rows.moment >= "2024-07-01"].moment.unique())
print(f"{'list':<12}{'vs training':>12}{'vs last list':>14}")
for before, mark in zip(marks, marks[1:]):
    now = rows[rows.moment == mark]
    last = Baseline(rows[rows.moment == before], ["supplier_voss"])
    print(f"{mark:%Y-%m-%d}{base.column(now, 'supplier_voss'):>12.2f}"
          f"{last.column(now, 'supplier_voss'):>14.2f}")
