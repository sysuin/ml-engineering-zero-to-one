# A simulation: a two-model uplift sketch, learned from randomised
# calls across each cohort's top 80, against v0.6's plain top 40.
# timeout: 300
import numpy as np

from foresight.config import rng
from foresight.impact.lists import lists, scored
from foresight.impact.simulate import World, fresh, true_chance
from foresight.impact.uplift import TwoModel, features

top80 = lists(scored("2025-01-01", "2025-12-31"), k=80)
p = true_chance(top80.contract_id)           # the simulator's truth
X = features(top80.chance)
month = top80.moment.rank(method="dense").astype(int).to_numpy()
half = month % 2                   # learn on one half, choose on other
groups = list(top80.groupby("moment").indices.values())
g = rng()


def choose(score):
    """Each cohort's 40 of 80 with the highest score."""
    pick = np.zeros(len(top80), bool)
    for idx in groups:
        pick[idx[np.argsort(-score[idx], kind="stable")[:40]]] = True
    return pick


def sketch(world, years):
    """Expected saves from the 40 the sketch picks, after `years` of
    randomised calls: 40 of each top 80 called, 40 held out."""
    rows, left, called = [], [], []
    for _ in range(years):
        c = np.zeros(len(top80), bool)
        for idx in groups:
            c[g.choice(idx, 40, replace=False)] = True
        rows.append(np.arange(len(top80)))
        left.append(fresh(p, c, world, g))
        called.append(c)
    rows, left = np.concatenate(rows), np.concatenate(left)
    called = np.concatenate(called)
    estimate = np.zeros(len(top80))
    for h in (0, 1):
        learn = half[rows] != h
        m = TwoModel().fit(X[rows[learn]], left[learn], called[learn])
        estimate[half == h] = m.predict(X[half == h])
    return world.uplift(p)[choose(estimate)].sum()


print("Expected saves a year, choosing 40 of each 2025 top 80")
for name, world in [("flat", World(save=0.25)),
                    ("further gone", World(save=0.7, gone=0.6))]:
    u = world.uplift(p)
    listed = u[top80["rank"].to_numpy() <= 40].sum()
    print(f"\n{name + ' world':<38}{'saves':>6}")
    print(f"  {"v0.6's top 40":<36}{listed:>6.1f}")
    print(f"  {'the truly most saveable (simulator)':<36}"
          f"{choose(u) @ u:>6.1f}")
    for years in (1, 10):
        got = np.array([sketch(world, years) for _ in range(100)])
        label = f"sketch, {years} year{'s' * (years > 1)} of calls"
        print(f"  {label:<36}{got.mean():>6.1f}"
              f"   beats v0.6 in {(got > listed).mean():.0%}")
