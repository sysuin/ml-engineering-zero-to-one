# The holdout for 2025's twelve lists: drawn within each cohort,
# checked for balance, and written once.
from foresight.config import DATA
from foresight.impact.holdout import AlreadyAssigned, assign, record
from foresight.impact.lists import lists, scored

listed = lists(scored("2025-01-01", "2025-12-31"))
drawn = assign(listed)

march = drawn[drawn.moment == "2025-03-02"]
print("Cohort marked 2025-03-02, the first ten of its list")
for r in march.head(10).itertuples():
    print(f"  {r.rank:>2}  contract {r.contract_id:<6} chance"
          f" {r.chance:5.1%}  {r.arm}")

print("\nAll twelve cohorts, by arm")
by_arm = drawn.groupby("arm").agg(
    contracts=("chance", "size"), mean_chance=("chance", "mean"),
    mean_rank=("rank", "mean"), top_ten=("rank", lambda r: (r <= 10)
                                         .sum()))
print(by_arm.round(3).to_string())
per_cohort = drawn.groupby("moment").arm.apply(
    lambda a: (a == "held out").sum())
print(f"held out in each cohort: {per_cohort.min()} to"
      f" {per_cohort.max()}")

replay = DATA / "foresight" / "holdout_2025_replay.csv"
replay.unlink(missing_ok=True)            # a replay, rebuilt each run
print(f"\nwritten: {record(drawn, replay)} rows;"
      f" again: {record(drawn, replay)} rows")
try:
    record(assign(listed, seed=1), replay)
except AlreadyAssigned as e:
    print(f"a second draw: {e}")
