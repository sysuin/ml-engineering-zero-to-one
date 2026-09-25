# The top 40 of each cohort as a confusion matrix, then in dollars.
import json

from foresight.costs import CostMatrix
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       load, top_of_each_cohort)

train, valid = load(*TRAIN), load(*VALIDATION)
valid["p"] = RenewalRisk().fit(train).predict_proba(valid)
called = valid.contract_id.isin(top_of_each_cohort(valid, valid.p)
                                .contract_id)
left = valid.not_renewed == 1
cells = {"tp": called & left, "fp": called & ~left,
         "fn": ~called & left, "tn": ~called & ~left}
n = {k: int(v.sum()) for k, v in cells.items()}
costs = CostMatrix(value_at_stake=2854, save_rate=0.25,
                   call_hours=1.5, hour_cost=60)
price = costs.cells()

print(f"Validation, top 40 of each cohort called: {called.sum()} calls")
print(f"  {'':12}{'left':>18}{'renewed':>20}")
for row, a, b in (("called", "tp", "fp"), ("not called", "fn", "tn")):
    print(f"  {row:12}" + "".join(
        f"{k.upper():>6}{n[k]:>7,}{price[k] or 0:>+7.0f}".replace(
            "+0", " 0") for k in (a, b)))
net = costs.net_value(n["tp"], n["fp"])
print(f"Net: {n['tp']} x ${price['tp']:.2f} - {n['fp']} x"
      f" ${-price['fp']:.0f} = ${net:,.0f} in six months")
print(f"Missed leavers forgo {n['fn']} x ${price['tp']:.2f}"
      f" = ${n['fn'] * price['tp']:,.0f}")

total = len(valid)
right = n["tp"] + n["tn"]
print(f"\nAccuracy  (TP + TN) / all   {right / total:.1%}")
print(f"  calling nobody             {(~left).mean():.1%}")
print(f"Precision TP / (TP + FP)    {n['tp'] / called.sum():.1%}")
print(f"Recall    TP / (TP + FN)    {n['tp'] / left.sum():.1%}")

print("\nSeptember's cohort: the highest score among the calls,"
      " the lowest\namong the rest")
print(f"  {'':4}{'contract':>8}  {'segment':<16}"
      f"{'days since order':>16}{'p':>8}")
sep = valid[valid.moment == "2024-09-01"]
for k, pick in (("tp", -1), ("fp", -1), ("fn", 0), ("tn", 0)):
    r = sep[cells[k][sep.index]].sort_values("p").iloc[pick]
    print(f"  {k.upper():4}{r.contract_id:>8}  {r.segment:<16}"
          f"{int(r.days_since_order):>16}{r.p:>8.2%}")

with open("code/07/07_confusion_as_money.json", "w") as f:
    json.dump({"n": n, "price": price, "net": net,
               "break_even": costs.break_even()}, f)
