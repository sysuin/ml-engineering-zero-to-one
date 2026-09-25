# Exercise 1: size a holdout for a team that can make forty calls a
# week, 160 a monthly cohort, from v0.6's validation lists.
import numpy as np

from foresight.impact.analyse import cohorts_needed, sample_size
from foresight.impact.lists import lists, scored

valid = lists(scored("2024-07-01", "2024-12-31"), k=320)
save = 0.25                                   # the brief's assumption
print(f"{'list':<9}{'left':>7}{'per arm':>9}  cohorts, held out of"
      " the list")
for k in (40, 160, 320):
    p0 = valid[valid["rank"] <= k].not_renewed.mean()
    n = sample_size(p0, p0 * (1 - save))
    cells = [np.ceil(cohorts_needed(p0, p0 * (1 - save), h, k - h))
             for h in (k // 2, k // 4, k // 8)]
    print(f"top {k:<5}{p0:>7.1%}{n:>9,.0f}  "
          + "  ".join(f"{h:>3}: {c:>3.0f}" for h, c in
                      zip((k // 2, k // 4, k // 8), cells)))
