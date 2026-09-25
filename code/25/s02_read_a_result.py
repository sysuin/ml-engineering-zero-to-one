# Exercise 2: a supplied result to read. SIMULATED: two years of the
# test, 2024 and 2025, in a world the reader is not told.
from foresight.config import SEED
from foresight.impact.analyse import compare, page
from foresight.impact.holdout import assign
from foresight.impact.lists import lists, scored
from foresight.impact.simulate import World, run

listed = lists(scored("2024-01-01", "2025-12-31"))
arm = assign(listed).set_index("contract_id").arm
listed["arm"] = arm.reindex(listed.contract_id).to_numpy()
called = listed.arm.eq("called").to_numpy()
world = World(save=0.7, gone=0.6)        # the further-gone world
left = run(listed, called, world, SEED)
record = listed[["moment", "contract_id", "arm"]].assign(left=left)
print(page(compare(record), "Retention test: 24 cohorts,"
           " Nov 2023 to Oct 2025"))
