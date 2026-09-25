# Gini and entropy, by hand, for one real split of the training rows.
import json

import numpy as np

from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
gap = features(train).days_since_order.to_numpy()  # none: 365
y = train.not_renewed.to_numpy()


def gini(y):
    """The chance that two contracts drawn from the node, with
    replacement, have different outcomes."""
    p = y.mean()
    return 1 - p ** 2 - (1 - p) ** 2


def entropy(y):
    """The bits needed, on average, to say how a contract ended."""
    p = y.mean()
    return -sum(q * np.log2(q) for q in (p, 1 - p) if q > 0)


def row(name, y):
    print(f"  {name:<18}{len(y):>6,}{y.sum():>8}{y.mean():>7.1%}"
          f"{gini(y):>8.4f}{entropy(y):>9.4f}")


def weighted(y, left, measure):
    """Each side's impurity, weighted by its share of the rows."""
    n = len(y)
    return (left.sum() / n * measure(y[left])
            + (~left).sum() / n * measure(y[~left]))


print(f"  {'':<18}{'rows':>6}{'leavers':>8}{'rate':>7}{'Gini':>8}"
      f"{'entropy':>9}")
row("training split", y)
for cut, whose in ((60, "the whiteboard's"), (92.5, "the tree's")):
    left = gap <= cut
    print(f"Split at {cut:g} days since the last order, {whose}")
    row(f"{cut:g} days or fewer", y[left])
    row(f"more than {cut:g}", y[~left])
    g, h = weighted(y, left, gini), weighted(y, left, entropy)
    print(f"  {'weighted':<39}{g:>8.4f}{h:>9.4f}")
    print(f"  {'decrease':<39}{gini(y) - g:>8.4f}"
          f"{entropy(y) - h:>9.4f}")

# Every cut between two observed values, scored both ways.
values = np.unique(gap)
cuts = (values[:-1] + values[1:]) / 2
drop = {name: np.array([m(y) - weighted(y, gap <= c, m)
                        for c in cuts])
        for name, m in (("Gini", gini), ("entropy", entropy))}
for name, d in drop.items():
    print(f"Best cut by {name}: {cuts[d.argmax()]:g} days,"
          f" decrease {d.max():.4f}")

left = gap <= 92.5
with open("code/10/02_impurity.json", "w") as f:
    json.dump({"parent": [len(y), int(y.sum()), gini(y)],
               "sides": [[int(s.sum()), int(y[s].sum()), gini(y[s])]
                         for s in (left, ~left)],
               "cuts": cuts.tolist(),
               "gini": drop["Gini"].tolist(),
               "entropy": drop["entropy"].tolist()}, f)
