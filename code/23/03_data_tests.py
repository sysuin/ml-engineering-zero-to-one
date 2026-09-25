# The data layer's tests on the training table: its schema against its
# manifest, whether it was built from the warehouse and the builder
# that are here now, its row count counted a second way, Chapter 4's
# checks, and each of Chapter 5's expectations as a test of its own.
from _suite import run

t = run("tests/test_data.py")
for name, g in t.groupby("name", sort=False):
    out = g.outcome.value_counts()
    said = ", ".join(f"{n} {o}" for o, n in out.items())
    print(f"{name.removeprefix('test_'):<52}{said:>14}")
print(f"\n{(t.outcome == 'passed').sum()} of {len(t)} pass")
