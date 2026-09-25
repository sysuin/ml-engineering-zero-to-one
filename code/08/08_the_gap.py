# When was each training outcome known, and what did the model know?
import json

import pandas as pd

from foresight.data.build_table import TABLE, build
from foresight.evaluate import SPLITS, backtest, known_by, measure
from foresight.models.logistic import (TRAIN, RenewalRisk,
                                       days_since_rule)

table = pd.read_parquet(TABLE)
train = table[table.end_date.between(*TRAIN)]
valid = table[table.end_date.between(*SPLITS["validation"])]

last = train.end_date.max()
print(f"Last training outcome on record: {last:%Y-%m-%d}")
print("Outcomes a model was fitted on that were not yet on record")
print(f"{'cohort ending':<15}{'its mark':>11}{'training split':>16}"
      f"{'earlier cohorts':>17}")
for mark, c in valid.groupby("moment"):
    fixed = (train.end_date >= mark).sum()
    earlier = (table[table.moment < mark].end_date >= mark).sum()
    month = f"{c.end_date.iloc[0]:%b %Y}"
    day = f"{mark:%Y-%m-%d}"
    print(f"{month:<15}{day:>11}{fixed:>16,}{earlier:>17,}")

# The builder can make the table as it stood on a day; the backtest's
# filter must give the same rows.
first = valid.moment.min()
built = build(*TRAIN, known_by=first)
same = built.equals(known_by(train, first).reset_index(drop=True))
print(f"\nbuild(known_by={first:%Y-%m-%d}): {len(built):,} rows,"
      f"\n  identical to the backtest's known_by(): {same}")


def fitted_on(pick):
    """Score each validation cohort, fitting on pick(mark)."""
    parts = []
    for mark, c in valid.groupby("moment"):
        fit = pick(mark)
        parts.append(c.assign(model=RenewalRisk().fit(fit)
                              .predict_proba(c),
                              rule=days_since_rule(c), base=0.1))
    return pd.concat(parts)


ways = {"training split": fitted_on(lambda mark: train),
        "earlier cohorts": fitted_on(
            lambda mark: table[table.moment < mark]),
        "known by the mark": backtest(table, *SPLITS["validation"])}
print(f"\nValidation, fitted on{'AUC':>14}{'top-40 leavers':>16}"
      f"{'precision':>11}")
result = {}
for name, s in ways.items():
    m = measure(s)
    result[name] = [m["auc"]["model"], m["precision"]["model"]]
    print(f"  {name:<19}{m['auc']['model']:>12.3f}"
          f"{round(m['precision']['model'] * 240):>16}"
          f"{m['precision']['model']:>11.1%}")
print(f"  {'(the rule)':<19}{m['auc']['rule']:>12.3f}"
      f"{round(m['precision']['rule'] * 240):>16}"
      f"{m['precision']['rule']:>11.1%}")

# And on the 5,027 contracts of section 8.5, known outcomes only.
pool = table[table.end_date <= SPLITS["validation"][1]]
later = backtest(pool, "2023-09-30", SPLITS["validation"][1])
m = measure(later)
hits = round(m["precision"]["model"] * 40 * later.moment.nunique())
print(f"\nThe same {len(later):,} contracts, known outcomes only:"
      f"\n  AUC {m['auc']['model']:.3f}, top-40 leavers {hits},"
      f" precision {m['precision']['model']:.1%}")

with open("code/08/08_the_gap.json", "w") as f:
    json.dump({"validation": result, "same": m["auc"]["model"],
               "same_precision": m["precision"]["model"]}, f)
