# A simulation: the before-and-after comparison, against a random
# holdout, in a world where calls work and a world where they do not.
import json

from foresight.config import SEED
from foresight.impact.analyse import before_after, rates
from foresight.impact.holdout import assign
from foresight.impact.lists import lists, scored
from foresight.impact.simulate import World, run, saved

listed = lists(scored("2024-01-01", "2025-12-31"))
before = listed[listed.end_date < "2025-01-01"].assign(
    left=lambda d: d.not_renewed)             # 2024: nobody called
after = listed[listed.end_date >= "2025-01-01"].reset_index(drop=True)
arm = assign(after).set_index("contract_id").arm
after["arm"] = arm.reindex(after.contract_id).to_numpy()
called = after.arm.eq("called").to_numpy()
print("2024 lists, nobody called        "
      f"{before.left.mean():6.1%} left")
print("2025 lists, if nobody were called"
      f"{after.not_renewed.mean():6.1%} left (simulator only)")

out = {"before": before.left.mean()}
for rate in (0.25, 0.0):
    world = World(save=rate)
    everyone = after.assign(left=run(after, True, world, SEED))
    test = after.assign(left=run(after, called, world, SEED))
    r = rates(test).rate
    true = saved(after, called, test.left) / called.sum()
    print(f"\nAssumed save rate {rate:.2f}; points fewer leaving")
    print(f"  before and after, all 40 called {before.left.mean():6.1%}"
          f" - {everyone.left.mean():5.1%} ="
          f" {100 * before_after(before, everyone):+5.1f}")
    print(f"  holdout, 20 of 40 called        {r['held out']:6.1%}"
          f" - {r['called']:5.1%} ="
          f" {100 * (r['held out'] - r['called']):+5.1f}")
    print(f"  what the calls did (simulator){100 * true:+22.1f}")
    out[str(rate)] = {"after": everyone.left.mean(),
                      "held": r["held out"], "called": r["called"],
                      "true": true}
with open("code/25/02_before_after.json", "w") as f:
    json.dump(out, f)
