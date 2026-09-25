# A recommender learns from clicks on what it chose to show. Without
# exploration it never learns about anything else.
import numpy as np

from foresight.config import rng

g = rng()
ITEMS, SHELF, VIEWS, DAYS, LAUNCH = 200, 10, 2_000, 90, 30
appeal = g.beta(2, 60, size=ITEMS)          # true click rate if shown
new = np.arange(ITEMS - 20, ITEMS)          # 20 items launch on day 30
appeal[new[:3]] = np.sort(appeal)[-3:] * 1.1    # three of them great
first_shelf = g.choice(ITEMS - 20, SHELF, replace=False)  # last season


def run(policy: str):
    shows, clicks = np.zeros(ITEMS), np.zeros(ITEMS)
    shows[first_shelf], clicks[first_shelf] = 5_000, 150   # history
    quality, found = [], {}
    local = rng()                 # every policy sees the same luck
    for day in range(DAYS):
        live = np.arange(ITEMS if day >= LAUNCH else ITEMS - 20)
        if policy == "most clicked":
            score = clicks[live]
        elif policy == "sampled from belief":   # Thompson sampling
            score = local.beta(clicks[live] + 1,
                               shows[live] - clicks[live] + 30)
        else:                     # observed rate; unseen items score 0
            score = clicks[live] / np.maximum(shows[live], 1)
        pick = list(live[np.argsort(-score, kind="stable")[:SHELF]])
        if policy == "rate + 2 slots explored":
            rest = np.setdiff1d(live, pick[:-2])
            pick[-2:] = local.choice(rest, 2, replace=False)
        pick = np.array(pick)
        shows[pick] += VIEWS
        clicks[pick] += local.binomial(VIEWS, appeal[pick])
        quality.append(appeal[pick].mean())
        for item in set(pick) & set(new[:3]):
            found.setdefault(item, day - LAUNCH)
    best = np.sort(appeal[np.arange(ITEMS)])[-SHELF:].mean()
    return (np.mean(quality[LAUNCH:]) / best,
            len(set(pick) & set(np.argsort(-appeal)[:SHELF])),
            sorted(found.values()))


print(f"{ITEMS} items, a shelf of {SHELF},"
      f" {VIEWS:,} views a slot a day")
print(f"{'':24}{'shelf vs best':>14}{'top 10 on':>11}{'great new':>14}")
print(f"{'':24}{'days 30-90':>14}{'day 90':>11}{'shown after':>14}")
for policy in ("most clicked", "click rate",
               "rate + 2 slots explored", "sampled from belief"):
    ratio, top, found = run(policy)
    days = ", ".join(f"{d}" for d in found) + " d" if found else "never"
    print(f"{policy:24}{ratio:>14.0%}{top:>8}/10{days:>14}")
