# The account team's whiteboard flowchart against a learned tree.
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       days_since_rule, features, load)

train, valid = load(*TRAIN), load(*VALIDATION)


def whiteboard(rows):
    """No order in 60 days: call first. Otherwise, a support ticket in
    the last 90 days: call next. Otherwise, no call."""
    lapsed = rows.days_since_order.astype(float).fillna(365) > 60
    ticket = rows.tickets_90d > 0
    return (2 * lapsed + (~lapsed & ticket)).to_numpy()


flagged = whiteboard(valid) > 0
per_cohort = flagged.sum() / valid.moment.nunique()
print(f"Validation: {len(valid):,} contracts, {valid.moment.nunique()}"
      f" cohorts, {valid.not_renewed.sum()} leavers")
print(f"The whiteboard says call {flagged.sum():,} of them,"
      f" {per_cohort:.0f} a cohort;")
print(f"  {valid.not_renewed[flagged].mean():.1%} of those left,"
      f" against {valid.not_renewed.mean():.1%} of all contracts")

tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
tree.fit(features(train), train.not_renewed)
lists = {"whiteboard": whiteboard(valid),
         "tree, depth 3": tree.predict_proba(features(valid))[:, 1],
         "the rule": days_since_rule(valid)}
print(f"\n{'':15}{'AUC':>7}{'top-40 leavers':>16}{'precision':>11}")
for name, score in lists.items():
    top = at_capacity(valid, score)
    print(f"{name:15}{auc(valid.not_renewed, score):>7.3f}"
          f"{top['leavers']:>16}{top['precision']:>11.1%}")
