# Renewals table: what exploration found

Training split only: 5,041 of 11,200 rows, contracts ending 2023-01-01 to 2024-06-30. Validation and test labels were not looked at.

## Question

What in this table is related to a contract not renewing, and which values cannot be taken at face value?

## Findings

| | Finding | Checked by |
|---|---|---|
| 1 | 328 of 5,041 renewals were not renewed (6.5%); cohorts ran from 3.7% to 10.1% | label rate per split |
| 2 | Recency's box-plot outliers are the leavers: past 72 days, 27.2% left; otherwise 4.5% | recency range |
| 3 | Spend is skewed (skew 35; on a log scale 0.05); the 79 key-account rows hold 22% of it. Not capped | none: this split only |
| 4 | Small business leaves most (9.5%), enterprise least (1.9%); regions differ no more than chance allows (5.5% to 7.6%) | none: this split only |
| 5 | discount_pct is missing on 1,216 rows, exactly the legacy contracts, which left at 8.1% against 6.0%. Never fill it | missing iff legacy |
| 6 | 14 rows have no order before the mark, 12 of them key accounts marked before their record starts: their zeros mean no record | zeros agree with dates |
| 7 | Key-account spend covers under a year until 2024: its median rose from 29,943 to 541,572 | none yet: needs a column |
| 8 | Small business left at 9.0% before July 2023 and 9.9% after, inside the noise | none: Chapter 8 |

## Decisions

- Keep the discount gaps. `legacy_terms` is their indicator; filling with 0 or the mean erases what the gap says.
- Give models spend on a log scale. Keep the key accounts: their size is real, and one of them is the most expensive loss.
- Read the zeros of the no-order rows as missing, not as nothing.
- Report leavers by row and by money, separately. Halloway is one row and most of its region's spend.

## Open questions

- Key accounts' windows are short before 2024. Add a column for how much history each row has (Chapter 12)?
- Did the small-business rate move after July 2023? Too small to tell here (Chapter 8).
- Why do legacy contracts carry no discount? Confirm with whoever owns the contracts table.

## Checks

12 expectations in `foresight/data/expectations.py`, run on every table the builder writes; broken on this one: 0. Rerun: `python -m foresight.data.expectations`.
