# Foresight renewal risk: model card, v0.6

**What it does.** Once a month, for the contracts reaching their
90-day mark, it gives each long-tail contract a chance of not
renewing, ranks them, and lists the top 40 for the account team's
retention calls. Each call carries its chance, a mark if the chance is
below the break-even of 12.6%, and up to three reasons in words.

## Intended use

- Choosing the account team's 40 retention calls in each monthly
  cohort of long-tail renewals, and the order to make them in.
- Marking calls below the break-even, so that the head of the account
  team can decide whether to make them.
- Expected revenue at risk for the long tail, by region, with its
  simulated range (Chapter 14).

## Not for

- **Key accounts.** Of the 116 key-account contracts that
  ended in the history the model learns from, 1 left:
  Halloway Healthcare. The model has no leavers like them to learn
  from and extrapolates beyond anything it has seen; at its mark it
  gave Halloway a chance of 1.9e-07. Key accounts are listed apart,
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

v0.5's lasso (L1 strength 0.002 per contract) on Chapter 4's
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

## Evaluation: validation, 6 monthly cohorts

Rolling-origin backtest over contracts ending 2024-07-01 to 2024-12-31:
2,083 contracts, 122 leavers, 240 calls. Each
cohort is scored by a model fitted on the outcomes known at its mark.
Intervals are 95%, from 2,000 resamples of contracts within cohorts.

| | v0.6 | Days-since rule | Ceiling (true chances) |
|---|---|---|---|
| Leavers in the top 40s | 60 | 53 | 64 |
| Precision at 40 | 25.0% (19.2%-30.4%) | 22.1% (16.7%-28.3%) | 26.7% |
| AUC | 0.818 (0.780-0.857) | 0.762 (0.714-0.808) | 0.819 |
| Log loss | 0.1848 | | base rate 0.2233 |
| Calibration slope | 1.29 | | 1 |

The ceiling ranks each cohort by the generator's true chances, which
no model can know; it was measured once (Chapters 9 and 13).

## Where it is wrong: slices, validation

| Slice | Contracts | Leavers | Missed (95%) | Renewers called | Chance minus share left, points (95%) |
|---|---|---|---|---|---|
| segment: Enterprise | 132 | 1 | 100% (100% to 100%) | 0% | +1.0 (-0.7 to +1.9) |
| segment: Mid-market | 706 | 17 | 71% (47% to 92%) | 3% | +1.4 (+0.2 to +2.4) |
| segment: Public sector | 229 | 3 | 33% (0% to 100%) | 1% | +0.9 (-0.7 to +2.1) |
| segment: Small business | 1,016 | 101 | 48% (38% to 57%) | 17% | -1.7 (-3.5 to 0.0) |
| size: smallest | 519 | 60 | 32% (20% to 44%) | 22% | -2.2 (-4.9 to +0.2) |
| size: second | 473 | 29 | 55% (37% to 72%) | 10% | 0.0 (-2.2 to +2.0) |
| size: third | 532 | 19 | 68% (46% to 88%) | 4% | +0.7 (-0.9 to +2.2) |
| size: largest | 559 | 14 | 100% (100% to 100%) | 2% | +0.6 (-0.7 to +1.8) |
| population: key accounts | 37 | 0 |  | 0% | +1.2 (+0.8 to +1.6) |
| population: long tail | 2,046 | 122 | 51% (42% to 59%) | 9% | -0.2 (-1.2 to +0.7) |

## Calibration

The lasso's own chances are timid (slope 1.46): too low at the
top of the list. Platt's map moves the slope to 1.29. Within
segments the chances may not be calibrated: on these cohorts,
Mid-market contracts got 1.4 points more than the share that left.

## Explanations

The reasons come from each contract's parts of the log-odds: weight
times standardised value, for this linear model exactly its SHAP
values, scaled by the Platt map's slope. The parts that raise the
risk by at least 0.05 in log-odds become reasons; the two order
counts are read as one. A value that fewer than 20 training rows
reach, on a column the model uses, is flagged as rarely seen.

## Known issues and open reviews

- `tenure_days` carries the calendar (Chapter 13's leakage review):
  open. Harmless to the lasso, whose tenure weight is near zero.
- The two order counts share their credit differently from fit to
  fit: the weight on the last quarter's orders was +0.000 in May 2024
  and +0.488 in October. Reasons read them together for this
  reason.
- The largest quarter of accounts by spend: 14 of its
  14 validation leavers were not called.
- Mid-market: 71% of its leavers missed, against
  48% for small businesses; reserving calls to close the
  gap cost leavers overall (Chapter 16).
- Rarely seen values are flagged, not refused.

## Owners

- The model and this card: the analytics team (Dana).
- The list's use, and whether calls below the break-even are made:
  the head of the account team.
- The costs behind the break-even: finance.
- Reviewed at every milestone; calibration checked on every cohort
  once its outcomes are known (Chapter 24).

## Test year, read once for v0.6

Contracts ending 2025-01-01 to 2025-12-31: 4,076 contracts, 364
leavers, 480 calls; read once, after v0.6 was fixed.

| | v0.6 | Days-since rule | Ceiling |
|---|---|---|---|
| Leavers in the top 40s | 144 | 123 | 173 |
| AUC | 0.749 | 0.708 | |
| Log loss | 0.2706 (uncalibrated 0.2721) | | |
| Calibration slope | 0.86 (uncalibrated 1.11) | | 1 |

v0.6 minus the rule, precision at 40, paired: +4.4 points (+0.8 to +6.9).

Slices on the test year: Mid-market's leavers were missed
81% of the time (73% to 89%), as on validation. Its
chances were -0.5 points against the share that left
(-1.8 to +0.6): validation's over-estimate did not recur.
