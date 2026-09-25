# The gate CI runs on a pull request, played out on three branches.
# Each branch's model and main's (v0.6) are backtested on the same six
# cohorts, the latest whose outcomes were known on 1 January 2025.
import json

import pandas as pd

from foresight import gate
from foresight.data.build_table import TABLE
from foresight.models.logistic import CALLS

table = pd.read_parquet(TABLE)
main, cfg = gate.candidate()
as_of, n = cfg["data"]["as_of"], cfg["evaluation"]["cohorts"]
champion = gate.scores(main, table, as_of, n)

tidy, _ = gate.candidate()                  # a refactor: same model
stronger, _ = gate.candidate(["model.strength=0.02"])
no_region, _ = gate.candidate()
no_region.set_params(estimator__prepare__region="drop")
branches = {"a tidy-up of the code": tidy,
            "strength 0.002 -> 0.02": stronger,
            "region dropped": no_region}

verdicts = {}
for name, model in branches.items():
    v = gate.check(champion, gate.scores(model, table, as_of, n))
    verdicts[name] = v
    page = gate.page(v, CALLS * n).splitlines()
    short = v.outcome != "blocked"          # the verdict and the list
    print(f"branch: {name}")
    print("\n".join(page[:4] if short else page) + "\n")


class Registry:                             # stands in for the real one
    def register(self, model, manifest, card=None, reason=""):
        return 2


try:
    gate.register(verdicts["strength 0.002 -> 0.02"], Registry(),
                  stronger, {})
except gate.GateError as e:
    head, reasons = str(e).split(": ", 1)
    print(f"register(): {head}:", *reasons.split("; "), sep="\n  ")

v = verdicts["strength 0.002 -> 0.02"]
with open("code/23/08_ci_gate.json", "w") as f:
    json.dump({"outcome": v.outcome, "reasons": v.reasons,
               "precision": v.overall["precision"],
               "auc": v.overall["auc"], "calls": CALLS * n,
               "slices": v.slices[v.slices.judged].to_dict("records")},
              f, indent=1)
