# Part V's compute in core-hours a month: measure a job, multiply.
# nondeterministic: timing
import time

from foresight.data.build_table import build
from foresight.models.logistic import RenewalRisk

# One night's batch: rebuild the table, refit, score every contract.
start = time.perf_counter()
table = build("2023-01-01", "2025-12-31")
scores = RenewalRisk().fit(table).predict_proba(table)
night = time.perf_counter() - start
print(f"One batch run, {len(scores):,} contracts:"
      f" {night:.1f} seconds on one core")

HOURS_A_MONTH = 24 * 365 / 12
jobs = [  # (what, core-hours a month, where the figure comes from)
    ("nightly batch, 30 runs", 30 * night / 3600, "measured above"),
    ("daily drift check", 30 * night / 3600, "assumed: like batch"),
    ("monthly retrain", 1.0, "assumed: one core-hour"),
    ("API, 1 core, always on", HOURS_A_MONTH, "24 x 365 / 12"),
]
print(f"\n{'job':<24}{'core-hours':>11}  source")
for what, hours, source in jobs:
    print(f"{what:<24}{hours:>11.2f}  {source}")
total = sum(hours for _, hours, _ in jobs)
share = HOURS_A_MONTH / total
print(f"{'a month':<24}{total:>11.2f}  the API is {share:.1%}")
print("\nMonthly compute cost = core-hours x your provider's rate")
print("per core-hour, plus storage, network and the registry.")
