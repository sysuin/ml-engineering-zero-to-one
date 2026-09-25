# Model against rule on the same contracts: the difference, resampled.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import (SPLITS, backtest, bootstrap, difference,
                                interval, measure)
from foresight.models.logistic import (TRAIN, RenewalRisk,
                                       days_since_rule)

table = pd.read_parquet(TABLE)
train = table[table.end_date.between(*TRAIN)]
valid = table[table.end_date.between(*SPLITS["validation"])]
chapter7 = valid.assign(model=RenewalRisk().fit(train)
                        .predict_proba(valid),
                        rule=days_since_rule(valid),
                        base=train.not_renewed.mean())
out = {}
v03 = backtest(table, *SPLITS["validation"])
for name, scored in (("Chapter 7, training split", chapter7),
                     ("v0.3 backtest", v03)):
    m, draws = measure(scored), bootstrap(scored)
    print(name)
    for who in ("model", "rule"):
        v = m["precision"][who]
        lo, hi = interval(draws[("precision", who)])
        print(f"  {who:<6}{round(v * 240):>4} leavers, {v:.1%}"
              f"   interval {lo:.1%} to {hi:.1%}")
    d = difference(draws, "precision")
    lo, hi = interval(d)
    gap = m["precision"]["model"] - m["precision"]["rule"]
    print(f"  model - rule {gap * 100:+.1f} points,"
          f" paired interval {lo * 100:+.1f} to {hi * 100:+.1f}")
    alone = (draws[("precision", "model")].std() ** 2
             + draws[("precision", "rule")].std() ** 2) ** 0.5
    print(f"  spread of the difference: paired {d.std() * 100:.1f}"
          f" points, as if unpaired {alone * 100:.1f}")
    print(f"  resamples where the model is ahead {(d > 0).mean():.0%},"
          f" level {(d == 0).mean():.0%}, behind {(d < 0).mean():.0%}")
    a = difference(draws, "auc")
    lo, hi = interval(a)
    gap = m["auc"]["model"] - m["auc"]["rule"]
    print(f"  AUC, model - rule {gap:+.3f}, paired interval"
          f" {lo:+.3f} to {hi:+.3f}")
    out[name] = {
        "precision": {x: [m["precision"][x],
                          *interval(draws[("precision", x)])]
                      for x in ("model", "rule")},
        "difference": [m["precision"]["model"] - m["precision"]["rule"],
                       *interval(d)],
        "leavers": (d * 240).round().astype(int).value_counts()
                   .sort_index().to_dict()}

with open("code/08/12_paired_comparison.json", "w") as f:
    json.dump(out, f)
