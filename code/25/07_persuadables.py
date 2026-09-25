# A simulation: who a call can save, when accounts that are further
# gone are harder to save. Ranking by risk against ranking by saves.
import json

import numpy as np

from foresight.impact.lists import lists, scored
from foresight.impact.simulate import World, true_chance
from foresight.slices import key_accounts

s = scored("2025-01-01", "2025-12-31")
s = s[~s.account_id.isin(key_accounts())].copy()
s["p"] = true_chance(s.contract_id)          # the simulator's truth
flat, far = World(save=0.25), World(save=0.7, gone=0.6)
listed = lists(s).set_index("contract_id")["rank"]
v06 = s.contract_id.isin(listed.index).to_numpy()
leavers = v06 & (s.not_renewed == 1).to_numpy()
print(f"Further gone: a call keeps {far.save:.0%} of the barely"
      f" leaving, and none")
print(f"at {far.gone:.0%} or more. On 2025's listed leavers it keeps"
      f" {far.save_chance(s.p[leavers]).mean():.1%}.\n")
s["u_flat"], s["u_far"] = flat.uplift(s.p), far.uplift(s.p)


def top(col):
    """Each cohort's 40 highest in `col`."""
    return (s.sort_values(["moment", col], ascending=[True, False])
             .groupby("moment").head(40).index)


chosen = {"v0.6's list": s.index[v06], "the truly riskiest": top("p"),
          "the truly most saveable": top("u_far")}
print("Expected saves a year from 480 calls")
print(f"{'list of 40 a cohort':<30}{'flat':>8}{'further gone':>14}")
for name, idx in chosen.items():
    print(f"{name:<30}{s.loc[idx, 'u_flat'].sum():>8.1f}"
          f"{s.loc[idx, 'u_far'].sum():>14.1f}")
same = len(set(chosen["the truly riskiest"])
           & set(chosen["the truly most saveable"]))
print(f"The riskiest and the most saveable share {same} of 480"
      " places.\n")

band = s[v06].assign(rank=listed.reindex(s.contract_id[v06]).to_numpy())
print("v0.6's list by rank, further-gone world")
print(f"{'ranks':<8}{'true chance':>12}{'save chance':>13}"
      f"{'expected saves':>16}")
for b, g in band.groupby((band["rank"] - 1) // 10):
    print(f"{10 * b + 1:>2}-{10 * b + 10:<5}{g.p.mean():>12.1%}"
          f"{far.save_chance(g.p).mean():>13.1%}{g.u_far.sum():>16.1f}")
with open("code/25/07_persuadables.json", "w") as f:
    json.dump({"p": np.round(s.p, 4).tolist(), "listed": v06.tolist(),
               "saveable": s.index.isin(
                   chosen["the truly most saveable"]).tolist(),
               "save": far.save, "gone": far.gone}, f)
