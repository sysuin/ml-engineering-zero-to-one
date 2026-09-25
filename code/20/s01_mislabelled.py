# Exercise 1: suspect labels after the rubric; what the pattern misses.
import re

from foresight.triage.evaluate import desk_pairs
from foresight.triage.features import (TRANSIT, in_transit, relabel,
                                       tickets)

t = tickets()
t["category"] = relabel(t)
pairs = desk_pairs(t[t.opened_at < "2025-01-01"])
off = pairs[pairs.north != pairs.south].drop_duplicates("north_body")
print(f"{len(off)} pairs still disagree; the first ten")
for _, r in off.head(10).iterrows():
    print(f"  N {r.north[:8]:8} S {r.south[:8]:8} {r.north_body[:44]}")

loose = re.compile(r"crush|soak|transit|driver|torn|spill|split", re.I)
missed = t[t.body.str.contains(loose) & ~t.body.map(in_transit)]
glove = missed.body.str.contains("glove", case=False)
print(f"\nloose words, missed by the pattern: {len(missed)}")
print(f"  glove split, chemical on a hand (a safety "
      f"ticket): {glove.sum()}")
for body in missed[~glove].body.head(3):
    print(f"  other: {body[:56]}")

wider = re.compile(TRANSIT.pattern + r"|driver d\w+ the delivery",
                   re.I)
moved = t.body.str.contains(wider) & ~t.body.map(in_transit)
print(f"\nadding 'driver d... the delivery' catches {moved.sum()} more")
print(f"'split' alone would also catch {glove.sum()} safety tickets")
