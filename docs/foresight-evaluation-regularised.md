# Foresight evaluation: regularised renewal risk

```text
Split      validation: contracts ending 2024-07-01 to 2024-12-31
           6 monthly cohorts, 2,083 contracts, 122 leavers
Method     rolling-origin backtest: each cohort scored by a
           model fitted on the 4,355 to 6,065 outcomes
           known at its mark; rule and random on the same rows
Intervals  95%, 2,000 bootstrap resamples of contracts
           within each cohort

Verdict    Model and rule cannot be told apart at capacity.

                model                rule                   baseline
precision at 40 25.0 (19.2-30.4)     22.1 (16.7-28.3)            5.9
recall at 40    49.2 (40.3-57.3)     43.4 (35.5-52.3)           11.5
AUC             0.815 (0.776-0.854)  0.762 (0.714-0.808)       0.500
log loss        0.187 (0.164-0.210)  -                         0.223
Brier score     0.049 (0.042-0.057)  -                         0.055

Precision and recall in per cent. Baseline: a random
list; for log loss and Brier, the training base rate for
every contract (the rule gives no probability to score).

Model minus rule, paired
  precision at 40   +2.9 points (-0.8 to +5.8)
  recall at 40      +5.7 points (-1.6 to +11.2)
  AUC               +0.053 (+0.022 to +0.087)

cohort mark    contracts  leavers  model hits  rule hits
2024-05-02           358       25           9          8
2024-06-02           348       14           8          8
2024-07-02           318       21          12         10
2024-08-02           349       23           8          9
2024-09-01           346       18           9          8
2024-10-02           364       21          14         10
all                2,083      122          60         53

A perfect list would reach 50.8% precision.
Not measured here: calibration and value in dollars
(Chapter 14), performance by segment (Chapter 16).
```
