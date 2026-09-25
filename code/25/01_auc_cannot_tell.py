# A simulation: v0.6's AUC on the 2025 cohorts as they happened, and
# in worlds where every account on the lists was called.
from foresight.config import SEED
from foresight.evaluate import auc
from foresight.impact.lists import lists, scored
from foresight.impact.simulate import World, run, saved

s = scored("2025-01-01", "2025-12-31")
listed = s.contract_id.isin(lists(s).contract_id).to_numpy()

print(f"v0.6 on {len(s):,} contracts in 12 cohorts, {listed.sum()}"
      " on the lists")
print(f"{'assumed save rate':<22}{'saved':>6}{'left':>6}"
      f"{'on list':>9}{'AUC':>8}")
print(f"{'nobody called':<22}{'-':>6}{s.not_renewed.sum():>6}"
      f"{s.not_renewed[listed].sum():>9}"
      f"{auc(s.not_renewed, s.chance):>8.3f}")
for rate in (0.0, 0.25, 0.4):
    left = run(s, listed, World(save=rate), SEED)
    print(f"{f'all listed called, {rate:.2f}':<22}"
          f"{saved(s, listed, left):>6}{left.sum():>6}"
          f"{left[listed].sum():>9}{auc(left, s.chance):>8.3f}")
