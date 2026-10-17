# Foresight: the launch checklist

Written by `code/27/02_launch_checklist.py`, which tests every line a script can test against a copy of the system. A line marked *by hand* needs a person to tick it; a *no* is a known gap, written down.

## Before anyone relies on it

- [x] the brief names decision, baseline and sign-off (yes)
- [x] the table is refused if an expectation fails (yes)
- [x] leakage checks run on every training job (yes)
- [x] the model's card sits beside it in the registry (yes)
- [x] a person promoted it, and wrote down why (yes)
- [ ] a version to roll back to is in the registry (no)

## The night it runs alone

- [x] the job refuses a stale warehouse (yes)
- [x] a second run does nothing (yes)
- [x] the job's rows match the API's (skew test) (yes)
- [x] the API refuses a malformed request (yes)
- [x] with no model, the rule answers (yes)

## Once it is live

- [x] alert thresholds are constants, not tuned later (yes)
- [x] the job draws and records every list's holdout (yes)
- [ ] a failed run alerts a person by the morning (no)
- [?] the CI gate blocks a worse model (make test) (by hand)
- [?] someone who did not build it ran make demo (by hand)
- [?] the account team knows 'held out' means no call (by hand)
- [?] an owner is named for each part (by hand)
