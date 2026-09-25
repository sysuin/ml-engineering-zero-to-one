# The library as Foresight now holds it: every feature by group, one
# record in full, the enriched table, and the tests that guard it.
import subprocess
import sys
import textwrap

from foresight.features.build import ENRICHED, load
from foresight.features.registry import REGISTRY, feature, groups, names

for g in groups():
    line = ", ".join(names(g))
    print(textwrap.fill(line, 66, initial_indent=f"{g:<13}",
                        subsequent_indent=" " * 13))

f = REGISTRY["order_trend"]
print(f"\n{f.name}: group {f.group}, added {f.added},"
      f" reads {', '.join(f.reads)},\n  looks back {f.window} days,"
      f" {f.linear} for a linear model")
print(textwrap.fill(f.about, 66, initial_indent="  ",
                    subsequent_indent="  "))

try:                                    # a second definition
    feature("order_trend", "trends")(lambda ctx: None)
except ValueError as e:
    print(f"\nValueError: {e}")

table = load()
print(f"\n{ENRICHED.name}: {len(table):,} rows,"
      f" {len(table.columns)} columns,\n  {len(names())} of them from"
      f" the library")

run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p",
                      "no:cacheprovider", "tests/test_features.py"],
                     capture_output=True, text=True)
print(f"tests/test_features.py: "
      f"{run.stdout.strip().splitlines()[-1].split(' in ')[0]}")
