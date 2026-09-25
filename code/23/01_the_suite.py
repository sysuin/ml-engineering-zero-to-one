# Foresight's test suite, counted by layer: every test collected (none
# run), with the layer tests/conftest.py gives it and whether it needs
# the generated dataset. Chapter 27's tests arrive after this chapter.
import json

from _suite import LAYERS, run

LATER = ["tests/test_mcp_server.py", "tests/test_demo.py"]
tests = run("tests/", "--collect-only",
            *[f"--ignore={f}" for f in LATER])
files = tests.file.nunique()
print(f"{len(tests)} tests in {files} files\n")
print(f"{'layer':<10}{'fast':>7}{'slow':>7}{'total':>7}{'files':>7}")
counts = {}
for layer in LAYERS:
    t = tests[tests.layer == layer]
    counts[layer] = {"fast": int((~t.slow).sum()),
                     "slow": int(t.slow.sum())}
    print(f"{layer:<10}{(~t.slow).sum():>7}{t.slow.sum():>7}"
          f"{len(t):>7}{t.file.nunique():>7}")
print(f"{'all':<10}{(~tests.slow).sum():>7}{tests.slow.sum():>7}"
      f"{len(tests):>7}{files:>7}")

print("\nwhere the layers above unit come from:")
for layer in LAYERS[1:]:
    t = tests[tests.layer == layer].file.value_counts()
    parts = [f"{f.removeprefix('test_')} {n}" for f, n in t.items()]
    line = f"  {layer}: " + ", ".join(parts)
    while len(line) > 68:
        cut = line.rindex(", ", 0, 68)
        print(line[:cut + 1])
        line = "    " + line[cut + 2:]
    print(line)

with open("code/23/01_the_suite.json", "w") as f:
    json.dump(counts, f, indent=1)
