# The two runs CI makes: `make test` (every push) and `make test-all`
# (nightly), with what happened to every test, by layer. Chapter 27's
# tests are left out, as in listing 23.1.
# timeout: 600
from _suite import LAYERS, run

LATER = ["--ignore=tests/test_mcp_server.py",
         "--ignore=tests/test_demo.py"]
runs = {"make test": run("tests/", "-m", "not slow", *LATER),
        "make test-all": run("tests/", *LATER)}

ENDS = ["passed", "failed", "skipped", "xfailed", "xpassed"]
print(f"{'':<15}{'layer':<9}" + "".join(f"{e:>8}" for e in ENDS))
for name, r in runs.items():
    for i, layer in enumerate([*LAYERS, "all"]):
        t = r if layer == "all" else r[r.layer == layer]
        n = t.outcome.value_counts()
        print(f"{name if i == 0 else '':<15}{layer:<9}"
              + "".join(f"{n.get(e, 0):>8}" for e in ENDS))
    print()
slow = runs["make test-all"]
slow = slow[slow.slow & (slow.outcome != "passed")]
for t in slow.itertuples():
    print(f"{t.outcome}: {t.file}::{t.name}")
