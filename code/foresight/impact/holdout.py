"""
The holdout: the accounts on the list that nobody calls, on purpose.
Chapter 25 writes it; the monthly list (score.py) feeds it.

    python -m foresight.impact.holdout --as-of 2025-03-02

Each month, after the list is made and before anyone picks up a phone,
`assign()` draws `held` of the cohort's top `k` at random and marks
them held out; the rest are called. The draw is seeded by the one seed
and the cohort's mark, so running the month again draws the same
accounts, and one month's draw cannot disturb another's.

`record()` appends the month to the assignments file and refuses to
change a month already there. An assignment made after anyone has seen
who renewed is no experiment at all, so the file is written once and
only read after that.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import DATA, SEED
from foresight.models.logistic import CALLS

ASSIGNMENTS = DATA / "foresight" / "holdout.csv"
HELD = 20                 # of each cohort's 40: Appendix F's design
COLUMNS = ["moment", "contract_id", "account_id", "rank", "chance",
           "arm"]


class AlreadyAssigned(RuntimeError):
    """A cohort's holdout is on record and may not be drawn again."""


def cohort_seed(mark, seed: int = SEED) -> int:
    """One seed per cohort: the book's seed plus the mark's date."""
    return seed + int(pd.Timestamp(mark).strftime("%Y%m%d"))


def assign(listed: pd.DataFrame, held: int = HELD,
           seed: int = SEED) -> pd.DataFrame:
    """Hold out `held` contracts of each cohort's list at random.
    `listed` has one row per contract on a list: its cohort's mark
    (`moment`), `contract_id`, `account_id`, `rank` and `chance`."""
    out = []
    for mark, group in listed.groupby("moment", sort=True):
        group = group.sort_values("rank")
        g = np.random.default_rng(cohort_seed(mark, seed))
        pick = g.choice(len(group), size=min(held, len(group)),
                        replace=False)
        arm = np.full(len(group), "called", dtype=object)
        arm[pick] = "held out"
        out.append(group.assign(arm=arm)[COLUMNS])
    return pd.concat(out, ignore_index=True)


def load(path: Path = ASSIGNMENTS) -> pd.DataFrame:
    """Every assignment on record, or an empty frame."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["moment"])


def record(assigned: pd.DataFrame, path: Path = ASSIGNMENTS) -> int:
    """Append new cohorts to the file; a cohort already there must
    match it exactly. Returns the number of rows written."""
    path = Path(path)
    have = load(path)
    marks = set(pd.to_datetime(have.moment))
    new = []
    for mark, group in assigned.groupby("moment"):
        if mark not in marks:
            new.append(group)
            continue
        old = have[pd.to_datetime(have.moment) == mark]
        a = old.set_index("contract_id").arm.sort_index()
        b = group.set_index("contract_id").arm.sort_index()
        if not a.equals(b):
            raise AlreadyAssigned(
                f"cohort {mark:%Y-%m-%d} is on record with"
                " another draw")
    if not new:
        return 0
    rows = pd.concat(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False,
                date_format="%Y-%m-%d")
    return len(rows)


def from_list(result: dict) -> pd.DataFrame:
    """The rows assign() needs, from one score.score() result."""
    listed = result["list"]
    return listed.assign(moment=result["mark"])[
        ["moment", "contract_id", "account_id", "rank", "chance"]]


def main(argv=None) -> None:
    from foresight.score import score
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--held", type=int, default=HELD)
    args = ap.parse_args(argv)
    result = score(args.as_of, k=CALLS)
    assigned = assign(from_list(result), args.held)
    written = record(assigned)
    counts = assigned.arm.value_counts()
    print(f"cohort {result['mark']:%Y-%m-%d}: "
          f"{counts.get('called', 0)} to call, "
          f"{counts.get('held out', 0)} held out; "
          f"{written} rows written to {ASSIGNMENTS}")


if __name__ == "__main__":
    main()
