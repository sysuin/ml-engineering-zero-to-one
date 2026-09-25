# A simulation: one year of the retention test, read from what
# Meridian would record, under four assumptions about what a call does.
import json

import numpy as np

from foresight.config import SEED, rng
from foresight.impact.analyse import HELD, compare, p_text, page
from foresight.impact.holdout import assign
from foresight.impact.lists import lists, scored
from foresight.impact.simulate import (World, fresh, run, saved,
                                       true_chance)

listed = lists(scored("2025-01-01", "2025-12-31"))
arm = assign(listed).set_index("contract_id").arm
listed["arm"] = arm.reindex(listed.contract_id).to_numpy()
called = listed.arm.eq("called").to_numpy()


def recorded(world):
    """What Meridian's warehouse holds after a year: list, arm, left."""
    left = run(listed, called, world, SEED)
    return listed[["moment", "contract_id", "arm"]].assign(left=left)


main = compare(recorded(World(save=0.25)))
print(page(main, "SIMULATED, assumed save rate 0.25: twelve cohorts"))
print(f"{'':2}would-be leavers in the called arm (simulator):"
      f" {listed.not_renewed[called].sum()}, saved"
      f" {saved(listed, called, recorded(World(0.25)).left)}\n")

print(f"{'assumed':<9}{'difference, points':>24}{'p':>9}"
      f"{'save rate':>11}{'kept':>7}")
rows = {}
for rate in (0.0, 0.1, 0.25, 0.4):
    rec = recorded(World(save=rate))
    r = compare(rec)
    true = saved(listed, called, rec.left) / listed.not_renewed[
        called].sum()
    rows[rate] = r
    print(f"{rate:<9.2f}{100 * r['diff']:>+7.1f} ({100 * r['lo']:+.1f}"
          f" to {100 * r['hi']:+.1f}){p_text(r['p_value']):>9}"
          f"{r['save_rate']:>11.0%}{true:>7.0%}")

# Many more years like this one, drawn from the true chances.
p = true_chance(listed.contract_id)
g, est, sig = rng(), [], []
for _ in range(2000):
    held = np.zeros(len(listed), bool)
    for idx in listed.groupby("moment").indices.values():
        held[g.choice(idx, 20, replace=False)] = True
    left = fresh(p, ~held, World(0.25), g)
    d = left[held].mean() - left[~held].mean()
    q = left.mean()
    se = np.sqrt(q * (1 - q) * (2 / 240))
    est.append(d)
    sig.append(abs(d) / se > 1.96)
est, sig = np.array(est), np.array(sig)
print(f"\n2,000 simulated years at 0.25: {sig.mean():.0%} significant;"
      f" average\ndifference {100 * est.mean():.1f} points,"
      f" {100 * est[sig].mean():.1f} among the significant ones")
numbers = {k: v for k, v in main.items() if k != "rates"}
numbers |= {"held": main["rates"].rate[HELD], "n": int(called.sum()),
            "sweep": {str(k): {x: v[x] for x in
                               ("diff", "lo", "hi", "p_value")}
                      for k, v in rows.items()}}
with open("code/25/06_the_test.json", "w") as f:
    json.dump(numbers, f, default=float)
