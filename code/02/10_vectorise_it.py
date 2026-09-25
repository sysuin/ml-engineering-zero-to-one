# The same 2025 revenue computed four ways, and timed.
# nondeterministic: timing
import json
import math
import sqlite3
import time
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
lines = pd.read_sql("SELECT l.qty, l.unit_price, l.discount_pct"
                    " FROM order_lines l JOIN orders o"
                    " USING (order_id)"
                    " WHERE o.order_date >= '2025-01-01'", con)
con.close()
columns = ("qty", "unit_price", "discount_pct")
qty, price, disc = (lines[c].tolist() for c in columns)  # lists
q, p, d = (lines[c].to_numpy() for c in columns)         # arrays


def python_loop():
    total = 0.0
    for i in range(len(qty)):
        total += qty[i] * price[i] * (1 - disc[i] / 100)
    return total


def pandas_apply():             # looks like pandas, runs like a loop
    return lines.apply(lambda r: r["qty"] * r["unit_price"]
                       * (1 - r["discount_pct"] / 100), axis=1).sum()


def pandas_columns():
    keep = 1 - lines["discount_pct"] / 100
    return (lines["qty"] * lines["unit_price"] * keep).sum()


def numpy_arrays():
    return float((q * p * (1 - d / 100)).sum())


def best_of(fn, repeat):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, min(times)


print(f"{len(lines):,} order lines in 2025\n")
timings = {}
for fn, repeat in [(python_loop, 3), (pandas_apply, 1),
                   (pandas_columns, 5), (numpy_arrays, 5)]:
    total, seconds = best_of(fn, repeat)
    timings[fn.__name__] = seconds
    ms = seconds * 1000
    print(f"  {fn.__name__:15} {total:>17,.2f}   {ms:9.1f} ms")

ratio = timings["python_loop"] / timings["numpy_arrays"]
print(f"\nthe loop took {ratio:,.0f} times as long as NumPy, this run")
print("loop and NumPy equal to 1e-12?",
      math.isclose(python_loop(), numpy_arrays(), rel_tol=1e-12))

# One run on one machine, saved for the chapter's figure.
Path(__file__).with_suffix(".json").write_text(json.dumps(
    {"rows": len(lines), "seconds": timings}, indent=1))
