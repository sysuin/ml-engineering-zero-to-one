# Three slips planted in Chapter 4's builder, one at a time. For each,
# the tiny-warehouse tests run, and the real table is rebuilt and put
# through Chapter 4's checks and Chapter 5's expectations: which of
# them notice? The builder's file is not touched; each slip replaces
# one function in memory while it lasts.
import inspect

import pandas as pd
from _suite import run

from foresight.data import build_table
from foresight.data.expectations import failures

SLIPS = [   # (the function, the text that is right, the slip)
    ("window_features", "o = o[o.age >= 1]", "o = o[o.age >= 0]"),
    ("window_features", "t = t[(age >= 1) & (age <= 90)]",
     "t = t[(age >= 0) & (age <= 90)]"),
    ("segment_as_of", "allow_exact_matches=False",
     "allow_exact_matches=True"),
]
TESTS = ("tests/test_build_table.py", "-m", "not slow")


def planted(name, right, slip):
    """The function `name` with `right` replaced by `slip`."""
    source = inspect.getsource(getattr(build_table, name))
    assert right in source
    scope = vars(build_table).copy()
    exec(source.replace(right, slip), scope)
    return scope[name]


table = pd.read_parquet(build_table.TABLE)
clean = run(*TESTS)
print(f"as written: {(clean.outcome == 'passed').sum()} of"
      f" {len(clean)} pass")
for name, right, slip in SLIPS:
    original = getattr(build_table, name)
    setattr(build_table, name, planted(name, right, slip))
    try:
        r = run(*TESTS)
        real = build_table.build("2023-01-01", "2025-12-31")
        seen = (len(build_table.check(real, "2023-01-01", "2025-12-31"))
                + len(failures(real)))
        moved = (real.ne(table) & ~(real.isna() & table.isna())) \
            .any(axis=1).sum()
    finally:
        setattr(build_table, name, original)
    failed = r[r.outcome == "failed"]
    print(f"\n{name}: {slip}")
    print(f"  tiny warehouse: {len(failed)} of {len(r)} tests fail")
    print(f"  real table: {moved:,} row(s) changed, {seen} checks"
          " fail")
    for t in failed.itertuples():
        print(f"  {t.name.removeprefix('test_')}")
        said = t.line if t.line.startswith("assert") else t.message
        said = " ".join(said.split()).split("build_table.")[-1]
        print(f"    {said[:62]}")
