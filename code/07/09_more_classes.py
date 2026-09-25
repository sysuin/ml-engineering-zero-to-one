# Four classes: an account's segment, by one-vs-rest and by softmax.
import numpy as np
from sklearn.linear_model import LogisticRegression

from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, VALIDATION, features, load

train, valid = (load(*s).dropna(subset=["segment"])
                for s in (TRAIN, VALIDATION))
cols = [c for c in features(train).columns
        if not c.startswith("segment=")]
scaler = Standardiser().fit(features(train)[cols].to_numpy())
Z, Zv = (scaler.transform(features(d)[cols].to_numpy())
         for d in (train, valid))
names = sorted(train.segment.unique())
y, yv = train.segment.to_numpy(), valid.segment.to_numpy()

# One-vs-rest: four yes-or-no models, "this segment or not".
ovr = {s: LogisticRegression(C=np.inf, max_iter=10_000).fit(Z, y == s)
       for s in names}
raw = np.column_stack([ovr[s].predict_proba(Zv)[:, 1] for s in names])
# Softmax: one model, four scores, turned into shares of one.
soft = LogisticRegression(C=np.inf, max_iter=10_000).fit(Z, y)
prob = soft.predict_proba(Zv)

row = int(np.argmax(valid.spend_365.to_numpy()))
print(f"Contract {valid.contract_id.iloc[row]}"
      f" ({valid.segment.iloc[row]})")
print(f"  {'':16}{'one-vs-rest':>12}{'softmax':>9}")
for i, s in enumerate(names):
    print(f"  {s:16}{raw[row, i]:>12.1%}{prob[row, i]:>9.1%}")
print(f"  {'sum':16}{raw[row].sum():>12.1%}{prob[row].sum():>9.1%}")

print(f"\nValidation accuracy, {len(yv):,} contracts")
common = train.segment.mode()[0]
for name, guess in (("always " + common, np.full(len(yv), common)),
                    ("one-vs-rest", np.array(names)[raw.argmax(1)]),
                    ("softmax", np.array(names)[prob.argmax(1)])):
    print(f"  {name:24}{(guess == yv).mean():>7.1%}")
