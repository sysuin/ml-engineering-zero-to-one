# The exact versions behind this book's printed output, and its floors.
import platform
import re
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from foresight.config import ROOT, THREADS

# Each line of requirements.txt that names a package: name>=floor.
REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)(\[[a-z]+\])?>=([\d.]+)")
TOOLCHAIN = {"pytest", "ruff", "pypdf", "pdfplumber", "fonttools",
             "pillow"}

rows, tools = [], []
for line in (ROOT / "requirements.txt").read_text().splitlines():
    m = REQUIREMENT.match(line.strip())
    if not m:
        continue
    name, floor = m.group(1), m.group(3)
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = "not installed"
    (tools if name in TOOLCHAIN else rows).append(
        (name, floor, installed))

blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
print(f"Python {platform.python_version()}"
      f" ({platform.python_implementation()})")
print(f"{platform.platform(terse=True)} on {platform.machine()},"
      f" NumPy's maths library: {blas['name']}")
print(f"Threads for training: {THREADS}\n")
print(f"{'package':<18}{'at least':>10}{'this book ran':>16}")
for name, floor, installed in rows:
    print(f"{name:<18}{floor:>10}{installed:>16}")
print("\nThe repository's own toolchain:")
for i in range(0, len(tools), 3):
    print("  " + "   ".join(f"{n} {v}" for n, _, v in tools[i:i + 3]))
