#!/usr/bin/env python3
"""
Check that everything the book needs is in place, before you spend an afternoon
discovering it is not.

    python3 code/_preflight.py

Nothing here needs a network connection or an API key. The one chapter that calls a
hosted language model (Chapter 20) checks for its own key when it gets there, and never
prints it: the most it shows is a fingerprint, the last four characters and the length.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

CHECKS: list[tuple[str, bool, str]] = []

# What each part of the book first needs. Import name, and the part that needs it.
PACKAGES = [
    ("pandas", "Part I"), ("numpy", "Part I"), ("duckdb", "Part I"),
    ("matplotlib", "Part I"), ("sklearn", "Part II"), ("lightgbm", "Part II"),
    ("optuna", "Part III"), ("mlflow", "Part III"), ("shap", "Part III"),
    ("torch", "Part IV"), ("fastapi", "Part V"), ("pydantic", "Part V"),
]


def check(label: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((label, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("\nEnvironment")
    check("Python 3.12 or newer", sys.version_info >= (3, 12),
          ".".join(map(str, sys.version_info[:3])))

    for package, part in PACKAGES:
        try:
            module = importlib.import_module(package)
            check(f"{package} installed", True, getattr(module, "__version__", ""))
        except ImportError:
            check(f"{package} installed", False,
                  f"needed from {part}: pip install -r requirements.txt")
        except OSError as e:
            # A library that installed but cannot load a system part it needs: on macOS,
            # LightGBM without the OpenMP runtime. Appendix B has the fix.
            check(f"{package} installed", False,
                  f"installed but will not load ({str(e).splitlines()[0][:60]}); "
                  "see Appendix B")

    print("\nDataset")
    from foresight.config import ML_WAREHOUSE, ROOT, WAREHOUSE
    for label, path in (("Meridian warehouse present", WAREHOUSE),
                        ("the scaled ML warehouse present", ML_WAREHOUSE)):
        check(label, path.exists(),
              str(path.relative_to(ROOT)) if path.exists() else "run: make data")

    print("\nDeterminism")
    from foresight.config import SEED, rng
    try:
        same = list(rng(SEED).integers(0, 1_000_000, 5)) == list(rng(SEED).integers(0, 1_000_000, 5))
        check("a seeded generator repeats itself", same, f"seed {SEED}")
    except ImportError:
        check("a seeded generator repeats itself", False, "numpy is not installed")

    print("\nSecrets")
    env_file = ROOT / ".env"
    if env_file.exists():
        ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT,
                                 capture_output=True).returncode == 0
        check(".env is ignored by git", ignored,
              "" if ignored else "add '.env' to .gitignore before doing anything else")
    else:
        print("  note  no .env — nothing needs one until Chapter 20")

    failed = [label for label, ok, _ in CHECKS if not ok]
    print(f"\n  {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print(f"  Fix first: {failed[0]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
