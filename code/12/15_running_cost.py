# What each group of features costs to keep: the tables it reads, the
# code that defines it, how far back it looks, and what it must fit.
import inspect
import json
import re

import foresight.data.build_table as builder
from foresight.features import definitions
from foresight.features.registry import REGISTRY, groups, names

FITTED = {"behaviour": "k-means"}      # fitted at every mark

print(f"{'group':<15}{'features':>8}{'lines':>6}{'window':>8}  reads")
out = {}
for g in groups():
    fs = [REGISTRY[n] for n in names(g)]
    reads = sorted({r for f in fs for r in f.reads}) or ["the table"]
    code = {inspect.getsource(f.compute) for f in fs}
    for c in list(code):        # and the helpers they call
        for h in re.findall(r"\b(_[a-z_]+)\(", c):
            if h not in ("_count", "_sum", "_share"):   # shared
                code.add(inspect.getsource(getattr(definitions, h)))
    lines = sum(len(c.strip().splitlines()) for c in code)
    window = max(f.window for f in fs)
    fits = FITTED.get(g, "")
    also = f" + {fits}" if fits else ""
    print(f"  {g:<13}{len(fs):>8}{lines:>6}{window:>8}  "
          f"{', '.join(reads)}{also}")
    out[g] = {"features": len(fs), "reads": reads, "lines": lines,
              "window": window, "fits": bool(fits)}
print(f"\nChapter 4's builder, for scale:"
      f" {len(inspect.getsource(builder).splitlines())} lines\n"
      f"  for its sixteen columns and their checks")
with open("code/12/15_running_cost.json", "w") as f:
    json.dump(out, f)
