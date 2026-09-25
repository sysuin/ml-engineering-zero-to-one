# Near-identical tickets filed by the two desks: do their labels agree?
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from foresight.triage.evaluate import kappa
from foresight.triage.features import CATEGORIES, split, tickets, tokens

train, valid, _ = split(tickets())
seen = pd.concat([train, valid], ignore_index=True)    # never 2025
north = seen[seen.desk == "North desk"].reset_index(drop=True)
south = seen[seen.desk == "South desk"].reset_index(drop=True)

# Each North ticket's closest South ticket, by the words they share.
vec = TfidfVectorizer(tokenizer=tokens, lowercase=False,
                      token_pattern=None).fit(seen.body)
index = NearestNeighbors(n_neighbors=1, metric="cosine")
index.fit(vec.transform(south.body))
distance, nearest = index.kneighbors(vec.transform(north.body))
close = 1 - distance[:, 0] >= 0.9
pairs = pd.DataFrame({
    "north": north.category[close].to_numpy(),
    "south": south.category.iloc[nearest[close, 0]].to_numpy(),
    "north_p": north.priority[close].to_numpy(),
    "south_p": south.priority.iloc[nearest[close, 0]].to_numpy(),
    "north_body": north.body[close].to_numpy(),
    "south_body": south.body.iloc[nearest[close, 0]].to_numpy()})
print(f"{len(pairs):,} of {len(north):,} North tickets have a South "
      f"ticket with nearly\nthe same words (cosine 0.9 or more)\n")

matrix = pd.crosstab(pairs.north, pairs.south).reindex(
    index=CATEGORIES, columns=CATEGORIES, fill_value=0)
print("North's category (rows) against South's (columns)")
print(matrix.rename(columns=lambda c: c[:5]).to_string())
agree = (pairs.north == pairs.south).mean()
print(f"\ncategory: same label {agree:.1%}, kappa "
      f"{kappa(pairs.north, pairs.south):.3f}")
p_agree = (pairs.north_p == pairs.south_p).mean()
print(f"priority: same label {p_agree:.1%}, kappa "
      f"{kappa(pairs.north_p, pairs.south_p):.3f}")

cell = pairs[(pairs.north == "Quality") & (pairs.south == "Delivery")]
print(f"\nNorth said Quality and South Delivery: {len(cell)} pairs")
for _, r in cell.drop_duplicates("north_body").head(3).iterrows():
    print(f"  N: {r.north_body[:62]}\n  S: {r.south_body[:62]}")
with open("code/20/05_label_disagreement.json", "w") as f:
    json.dump({"categories": CATEGORIES,
               "matrix": matrix.to_numpy().tolist()}, f)
