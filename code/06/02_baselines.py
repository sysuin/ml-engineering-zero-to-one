# Three baselines for next quarter's spend, scored on validation.
from foresight.data.spend import (TRAIN, VALIDATION, baselines, mae,
                                  rmse, scores, spend_table)

train = spend_table(*TRAIN)
valid = spend_table(*VALIDATION)
y = valid.spend_next_90d.to_numpy()

print(f"Validation, {len(valid):,} contracts ending "
      f"{VALIDATION[0]} to {VALIDATION[1]}")
guesses = baselines(train, valid)
scores(y, guesses)

# The same three on the rows they were built from.
t = train.spend_next_90d.to_numpy()
print(f"\nTraining rows, {len(train):,}")
scores(t, baselines(train, train))
print()
for metric in (mae, rmse):
    best = min(guesses, key=lambda k: metric(y, guesses[k]))
    print(f"lowest validation {metric.__name__.upper()}: {best}")
