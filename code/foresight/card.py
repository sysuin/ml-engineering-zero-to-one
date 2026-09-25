"""
Foresight's model card: what the renewal model is for, what it is not
for, and how it was judged. Chapter 16 writes it.

The card is written by people and filled in by the build. Its words
live here, as a template; every number in it is measured by the
listing that renders it, so the card cannot drift from what the model
does. render() refuses a card with a number missing, and the test year
section is added once, by the milestone's single read of the test
year, and never rewritten.
"""
from __future__ import annotations

from pathlib import Path

from foresight.config import ROOT

CARD = ROOT / "docs" / "foresight-model-card.md"

TEMPLATE = """\
# Foresight renewal risk: model card, v{version}

**What it does.** Once a month, for the contracts reaching their
90-day mark, it gives each long-tail contract a chance of not
renewing, ranks them, and lists the top {calls} for the account team's
retention calls. Each call carries its chance, a mark if the chance is
below the break-even of {break_even}, and up to three reasons in words.

## Intended use

- Choosing the account team's {calls} retention calls in each monthly
  cohort of long-tail renewals, and the order to make them in.
- Marking calls below the break-even, so that the head of the account
  team can decide whether to make them.
- Expected revenue at risk for the long tail, by region, with its
  simulated range (Chapter 14).

## Not for

- **Key accounts.** Of the {key_contracts} key-account contracts that
  ended in the history the model learns from, {key_left} left:
  Halloway Healthcare. The model has no leavers like them to learn
  from and extrapolates beyond anything it has seen; at its mark it
  gave Halloway a chance of {halloway}. Key accounts are listed apart,
  unranked, for their account managers.
- **Deciding what to say or offer on a call.** A reason says what
  moved this model's score, not what would keep the account. A low
  discount among the reasons is not evidence that a discount would
  help.
- **Judging account managers**, pricing, credit, or any decision
  about a person.
- **Any other horizon or unit.** The chance is for a contract not
  renewing, decided at its 90-day mark.

## Model

v0.5's lasso (L1 strength {strength} per contract) on Chapter 4's
eleven columns, refitted each month on the outcomes known that
morning; its chances corrected by a Platt map learned on the latest
three cohorts the model may see. Checked by the leakage checks of
Chapter 13 on every fit.

## Data

One row per contract renewal, as it stood on the morning of its
90-day mark (`data/foresight/renewals_table.parquet`); label: the
contract was not renewed. History from contracts ending in January
2023; the key accounts' orders are on record from January 2023 only.
Outcomes are used only once they are on record.

## Evaluation: validation, {cohorts} monthly cohorts

Rolling-origin backtest over contracts ending {first} to {last}:
{contracts} contracts, {leavers} leavers, {calls_total} calls. Each
cohort is scored by a model fitted on the outcomes known at its mark.
Intervals are 95%, from {reps} resamples of contracts within cohorts.

| | v0.6 | Days-since rule | Best possible (ceiling) |
|---|---|---|---|
| Leavers in the top {calls}s | {hits} | {rule_hits} | {ceiling_hits} |
| Precision at {calls} | {precision} | {rule_precision} | {ceiling_precision} |
| AUC | {auc} | {rule_auc} | {ceiling_auc} |
| Log loss | {log_loss} | | base rate {base_log_loss} |
| Calibration slope | {slope} | | 1 |

The ceiling ranks each cohort by the generator's true chances, which
no model can know; it was measured once (Chapters 9 and 13).

## Where it is wrong: slices, validation

{slices}

## Calibration

The lasso's own chances are timid (slope {raw_slope}): too low at the
top of the list. Platt's map moves the slope to {slope}. Within
segments the chances are not calibrated: Mid-market contracts are
given {mid_gap} points more than the share that left.

## Explanations

The reasons come from each contract's parts of the log-odds: weight
times standardised value, for this linear model exactly its SHAP
values, scaled by the Platt map's slope. The parts that raise the
risk by at least {smallest} in log-odds become reasons; the two order
counts are read as one. A value that fewer than {rare} training rows
reach, on a column the model uses, is flagged as rarely seen.

## Known issues and open reviews

- `tenure_days` carries the calendar (Chapter 13's leakage review):
  open. Harmless to the lasso, whose tenure weight is near zero.
- The two order counts share their credit differently from fit to
  fit: the weight on the last quarter's orders was {w_may} in May 2024
  and {w_october} in October. Reasons read them together for this
  reason.
- The largest quarter of accounts by spend: {largest_missed} of its
  {largest_leavers} validation leavers were not called.
- Mid-market: {mid_missed} of its leavers missed, against
  {small_missed} for small businesses; reserving calls to close the
  gap cost leavers overall (Chapter 16).
- Rarely seen values are flagged, not refused.

## Owners

- The model and this card: the analytics team (Dana).
- The list's use, and whether calls below the break-even are made:
  the head of the account team.
- The costs behind the break-even: finance.
- Reviewed at every milestone; calibration checked on every cohort
  once its outcomes are known (Chapter 24).
"""

TEST = """
## Test year, read once for v{version}

Contracts ending {first} to {last}: {contracts} contracts, {leavers}
leavers, {calls_total} calls; read once, after v{version} was fixed.

| | v{version} | Days-since rule | Ceiling |
|---|---|---|---|
| Leavers in the top {calls}s | {hits} | {rule_hits} | {ceiling_hits} |
| AUC | {auc} | {rule_auc} | |
| Log loss | {log_loss} (uncalibrated {raw_log_loss}) | | |
| Calibration slope | {slope} (uncalibrated {raw_slope}) | | 1 |

v{version} minus the rule, precision at {calls}, paired: {diff}.
"""


def render(numbers: dict, test: dict | None = None) -> str:
    """The card, with every number filled in. A missing number raises
    KeyError: a card is not written with a gap in it."""
    text = TEMPLATE.format(**numbers)
    if test is not None:
        text += TEST.format(**test)
    return text


def write(numbers: dict, test: dict | None = None,
          path: Path = CARD) -> str:
    """Render the card and write it to `path`."""
    text = render(numbers, test)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
