# Categories as numbers: how many columns one-hot encoding needs, how
# thin they get, and an ordinal code where the order means something.
from foresight.features.build import load
from foresight.features.sources import load_sources
from foresight.models.featured import (TRAINING, FeaturedLasso,
                                       tuning_score)

table = load()
postcode = load_sources().accounts.set_index("account_id").postcode
table["area"] = table.account_id.map(postcode).str[:3]
table["postcode"] = table.account_id.map(postcode)
top = table.filter(like="share_").idxmax(axis=1)
top = top.str.removeprefix("share_").str.capitalize()
table["main_category"] = top.where(table.spend_365 > 0, "none")
train = table[table.end_date.between(*TRAINING)]

print(f"{'Training rows':<16}{'values':>7}{'columns':>9}"
      f"{'under 20 rows':>15}{'their rows':>12}")
for col in ["segment", "region", "main_category", "area", "postcode"]:
    n = train[col].value_counts()
    thin = n[n < 20]
    print(f"  {col:<14}{len(n):>7,}{len(n) - 1:>9,}{len(thin):>15,}"
          f"{thin.sum() / len(train):>12.1%}")

# Ordinal: one column, the segments in order of their median spend.
spend = train.groupby("segment").spend_365.median().sort_values()
rank = {s: i for i, s in enumerate(spend.index)}
print("\nSegments by median spend in the year before the mark")
for s, v in spend.items():
    left = train.not_renewed[train.segment == s].mean()
    print(f"  {rank[s]}  {s:<16}{v:>10,.0f} dollars{left:>9.1%} left")

table["segment_rank"] = table.segment.map(rank).fillna(0)


def make_ord():
    return FeaturedLasso(["segment_rank"], drop=("segment=",))


print(f"\n{'Tuning cohorts':<28}{'log loss':>9}{'AUC':>7}"
      f"{'leavers':>9}")
for name, make in (("one-hot segment, 3 columns", FeaturedLasso),
                   ("ordinal segment, 1 column", make_ord)):
    s = tuning_score(table, make)
    print(f"  {name:<26}{s['log loss']:>9.5f}{s['auc']:>7.3f}"
          f"{s['hits']:>9}")
print(f"\nOne-hot on every category above:"
      f" {sum(train[c].nunique() - 1 for c in ['segment', 'region',
             'main_category', 'area', 'postcode']):,} columns for"
      f" {len(train):,} rows")
