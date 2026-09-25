# Exercise 3: SIMULATED. Hold out one of each pair of neighbouring
# ranks, against twenty drawn from the whole forty, over 2,000 years.
import numpy as np

from foresight.config import rng
from foresight.impact.lists import lists, scored
from foresight.impact.simulate import World, fresh, true_chance

listed = lists(scored("2025-01-01", "2025-12-31"))
p = true_chance(listed.contract_id)
groups = list(listed.groupby("moment").indices.values())
g = rng()


def whole(idx):
    return g.choice(idx, 20, replace=False)


def pairs(idx):
    ranked = idx[np.argsort(listed["rank"].to_numpy()[idx])]
    return ranked.reshape(20, 2)[np.arange(20), g.integers(0, 2, 20)]


for name, draw in (("twenty of forty", whole),
                   ("one of each pair", pairs)):
    est = []
    for _ in range(2000):
        held = np.zeros(len(listed), bool)
        for idx in groups:
            held[draw(idx)] = True
        left = fresh(p, ~held, World(0.25), g)
        est.append(left[held].mean() - left[~held].mean())
    est = 100 * np.array(est)
    print(f"{name:<18} mean {est.mean():+.1f} points,"
          f" spread (sd) {est.std():.2f}")
