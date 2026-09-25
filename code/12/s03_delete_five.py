# Exercise 3: the lasso given the whole library, then without the five
# features it leans on least. What did the five cost?
from foresight.evaluate import SPLITS, backtest
from foresight.features.build import load
from foresight.features.registry import names
from foresight.models.featured import (SHOWN, TRAINING, gain, lasso,
                                       shown, tuning_score)

table = load()
w = lasso(names())().fit(
    table[table.end_date.between(*TRAINING)]).weights()[names()]
used = w[w != 0].abs().sort_values()
weakest = list(used.index[:5])
fewer = [f for f in names() if f not in weakest]
print(f"Library features the lasso uses: {len(used)} of {len(w)}")
print(f"Weakest five: {', '.join(weakest[:3])},\n  "
      f"{', '.join(weakest[3:])}")

print(f"\n{'Tuning cohorts':<24}{'columns':>8}{'log loss':>10}"
      f"{'AUC':>7}")
for name, extra in (("whole library", names()), ("without the five",
                                                  fewer)):
    s = tuning_score(table, lasso(extra))
    print(f"  {name:<22}{16 + len(extra):>8}{s['log loss']:>10.5f}"
          f"{s['auc']:>7.3f}")

V = SPLITS["validation"]
g = gain(backtest(table, *V, lasso(fewer)),
         backtest(table, *V, lasso(names())))
print(f"\nValidation, without minus with the five\n  {'':10}{SHOWN}")
print(f"  {'lasso':<13}{shown(g)}")
