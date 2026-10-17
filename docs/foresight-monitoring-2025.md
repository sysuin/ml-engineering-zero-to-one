# Foresight monitoring, 2025

Written by `python -m foresight.monitor.report`. Each page shows only
what was on record on its day.

## The week to 07 April 2025

```text
Foresight monitoring: the week to Mon 07 Apr 2025
Serving v0.6: 7,124 outcomes, fitted 2025-01-01
In shadow: monthly, voss share, exposure

Alerts this week
  2025-04-02  notices  list 2025-03-02   41.00  38.00 expected 21.9

The latest list, 2025-04-01
  contracts 353, usual 307
  chances: average 5.6%, PSI 0.04 against training
  inputs over PSI 0.25, with the drift's effect on the average
  log-odds:
    tenure_days         0.43  -0.002

Notices: the last four lists whose notices are in
  list         notices  expected    limits
  2025-01-30        19      15.1     5 to 28
  2025-03-02        41      21.9     9 to 38  !

Outcomes on record: 0 lists, 0 contracts

The gate, this month (challenger minus champion)
  monthly    0 lists with outcomes: too few to judge
  voss share 0 lists with outcomes: too few to judge
  exposure   0 lists with outcomes: too few to judge

Demand forecast, 2025-03, one month ahead
  WAPE 12.9% (limit 18.9%), seasonal naive 22.0%
  80% intervals held 86.8%; over six months 88.6%
```

## The week to 31 December 2025

```text
Foresight monitoring: the week to Wed 31 Dec 2025
Serving exposure: 10,051 outcomes, fitted 2025-10-02
In shadow: v0.6, monthly, voss share

Alerts this week
  none

The latest list, 2025-12-31
  contracts 283, usual 352
  chances: average 14.9%, PSI 0.13 against training
  inputs over PSI 0.25, with the drift's effect on the average
  log-odds:
    exposure            2.15  +0.435  acknowledged
    tenure_days         0.48  +0.008
    discount_pct        0.34  -0.047
    legacy_terms        0.33  -0.029

Notices: the last four lists whose notices are in
  list         notices  expected    limits
  2025-07-02        39      19.2     7 to 34  !
  2025-08-02        26      19.5     7 to 34
  2025-09-01        33      26.2    12 to 43
  2025-10-02        41      65.6    42 to 92  !

Outcomes on record: 8 lists, 2,843 contracts
  left 249, expected 162.4; calibration slope 0.74
  average chance 5.7%, share that left 8.8%
  leavers in the calls: list 99, rule 84, of 320

The gate, this month (challenger minus champion)
  monthly    -2.5 pts (-4.7 to +0.9), AUC -0.010: stays in shadow
  voss share -2.5 pts (-4.4 to +0.9), AUC -0.005: stays in shadow
  v0.6       -2.2 pts (-5.0 to +0.9), AUC -0.012: stays in shadow

Demand forecast, 2025-11, one month ahead
  WAPE 10.7% (limit 16.0%), seasonal naive 15.2%
  80% intervals held 91.6%; over six months 88.2%
```

## Every alert of the year

```text
2025-01-01  feed     Voss Industrial    1.12   1.00 cost, 4 series
2025-02-01  feed     Voss Industrial    1.08   1.00 price, 4 series
2025-04-02  notices  list 2025-03-02   41.00  38.00 expected 21.9
2025-06-01  outcomes list 2025-03-02   41.00  38.00 expected 21.9
2025-06-02  notices  list 2025-05-02   37.00  35.00 expected 19.9
2025-08-01  outcomes list 2025-05-02   37.00  35.00 expected 19.9
2025-08-02  notices  list 2025-07-02   39.00  34.00 expected 19.2
2025-10-01  outcomes list 2025-07-02   39.00  34.00 expected 19.2
2025-11-02  notices  list 2025-10-02   41.00  42.00 expected 65.6
```
