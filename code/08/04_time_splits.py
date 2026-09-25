# Train on earlier cohorts, score the next: a time split, by hand.
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, measure
from foresight.models.logistic import RenewalRisk, days_since_rule

table = pd.read_parquet(TABLE)
pool = table[table.end_date <= SPLITS["validation"][1]]
pool = pool.reset_index(drop=True)          # sorted by mark already


def by_cohort(rows, first_mark):
    """Each cohort from first_mark on is held out in turn, and the
    model is fitted on every cohort whose mark came before it."""
    for mark in sorted(rows.moment.unique()):
        if mark >= pd.Timestamp(first_mark):
            yield rows[rows.moment < mark], rows[rows.moment == mark]


print(f"{'cohort (mark)':<15}{'fit on':>8}{'scored':>8}{'leavers':>9}"
      f"{'model AUC':>11}{'rule AUC':>10}")
scored = []
for fit, cohort in by_cohort(pool, "2024-05-01"):
    p = RenewalRisk().fit(fit).predict_proba(cohort)
    y = cohort.not_renewed
    print(f"{cohort.moment.iloc[0]:%Y-%m-%d}{len(fit):>13,}"
          f"{len(cohort):>8}{y.sum():>9}{auc(y, p):>11.3f}"
          f"{auc(y, days_since_rule(cohort)):>10.3f}")
    scored.append(cohort.assign(model=p, rule=days_since_rule(cohort),
                                base=fit.not_renewed.mean()))
m = measure(pd.concat(scored))
print(f"Pooled AUC: model {m['auc']['model']:.3f},"
      f" rule {m['auc']['rule']:.3f}")
print(f"Top 40 a cohort: model {m['precision']['model']:.1%},"
      f" rule {m['precision']['rule']:.1%}")

# The library's time split knows row order, not cohorts.
cut = TimeSeriesSplit(n_splits=6)
marks = pool.moment.to_numpy()
torn = sum(marks[fit[-1]] == marks[out[0]]
           for fit, out in cut.split(pool))
print(f"TimeSeriesSplit: {torn} of 6 folds cut a cohort in two")

