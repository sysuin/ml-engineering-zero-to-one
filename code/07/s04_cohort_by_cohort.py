# Exercise 4: the model and the rule, one cohort at a time.
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       days_since_rule, load,
                                       top_of_each_cohort)

train, valid = load(*TRAIN), load(*VALIDATION)
p = RenewalRisk().fit(train).predict_proba(valid)
model = top_of_each_cohort(valid, p)
rule = top_of_each_cohort(valid, days_since_rule(valid))
shared = set(model.contract_id) & set(rule.contract_id)
print(f"  {'mark':<12}{'leavers':>8}{'model':>7}{'rule':>6}"
      f"{'on both lists':>15}")
for moment, cohort in valid.groupby("moment"):
    m = model[model.moment == moment].not_renewed.sum()
    r = rule[rule.moment == moment].not_renewed.sum()
    both = cohort[cohort.contract_id.isin(shared)].not_renewed.sum()
    print(f"  {moment:%Y-%m-%d}  {cohort.not_renewed.sum():>8}{m:>7}"
          f"{r:>6}{both:>15}")
