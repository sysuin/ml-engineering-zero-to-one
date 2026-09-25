# Floors for the model tests, set from v0.6's own backtest: the lower
# end of each 95% interval, overall and for every slice with at least
# ten leavers. Written to tests/model_floors.json for the tests to read.
import pandas as pd

from foresight import floors, gate
from foresight.data.build_table import TABLE
from foresight.evaluate import measure
from foresight.models.logistic import CALLS

table = pd.read_parquet(TABLE)
v06, cfg = gate.candidate()             # the settings file: v0.6
as_of, cohorts = cfg["data"]["as_of"], cfg["evaluation"]["cohorts"]
scored = gate.scores(v06, table, as_of, cohorts)
f = floors.set_floors(scored)
floors.write(f, {"reference": "v0.6", "as_of": as_of,
                 "cohorts": cohorts})

m = measure(scored.assign(base=0.0))
calls = CALLS * cohorts
print(f"v0.6 on the {cohorts} cohorts known on {as_of}, {calls} calls")
print(f"{'':24}{'v0.6':>9}{'rule':>9}{'floor':>9}")
for what, kind in (("precision", "{:.1%}"), ("auc", "{:.3f}")):
    print(f"{what:<24}" + "".join(
        f"{kind.format(v):>9}" for v in (m[what]["model"],
                                       m[what]["rule"], f[what])))
print(f"\n{'slice':<24}{'leavers':>9}{'called':>9}{'floor':>9}")
for by, kept in f["slices"].items():
    share = floors.called_share(scored, by)
    left = scored.groupby(by, observed=True).not_renewed.sum()
    for name, least in kept.items():
        print(f"{by + ' ' + name:<24}{left[name]:>9}"
              f"{share[name]:>9.1%}{least:>9.1%}")
none = [f"{by} {n}" for by in f["slices"]
        for n, x in f["slices"][by].items() if x == 0]
print("floor of zero:", ", ".join(none) or "none")
print("\nv0.6 against its own floors:",
      floors.shortfalls(scored, f) or "kept")

stronger, _ = gate.candidate(["model.strength=0.02"])
s = gate.scores(stronger, table, as_of, cohorts)
print("strength 0.02 against them:")
for line in floors.shortfalls(s, f):
    print("  " + line)
