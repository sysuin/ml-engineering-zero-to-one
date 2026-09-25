# From text to columns: tokens, a vocabulary, and a bag of words.
from sklearn.feature_extraction.text import CountVectorizer

from foresight.triage.features import split, tickets, tokens

train, _, _ = split(tickets())
three = train.set_index("ticket_id").loc[
    ["FT-000214", "FT-000232", "FT-000274"], "body"]
for tid, body in three.items():
    print(f"{tid}  {body}")
    print(f"{'tokens':>9}  " + ", ".join(tokens(body)))

bag = CountVectorizer(tokenizer=tokens, lowercase=False,
                      token_pattern=None)
X = bag.fit_transform(train.body)
words = bag.get_feature_names_out()
print(f"\n{len(words):,} different words in {len(train):,} training "
      f"tickets")
zeros = 1 - X.nnz / (X.shape[0] * X.shape[1])
used = X.getnnz(axis=1).mean()
print(f"a ticket uses {used:.1f} of them on average, so {zeros:.2%} "
      f"of the")
print("bag-of-words table is zeros")

show = ["twice", "inv", "sku", "stock", "asap", "missing", "num"]
cols = [bag.vocabulary_[w] for w in show]
rows = bag.transform(three).toarray()[:, cols]
print("\n" + " " * 10 + "".join(f"{w:>8}" for w in show))
for tid, r in zip(three.index, rows):
    print(f"{tid:10}" + "".join(f"{n:8d}" for n in r))

counts = X.sum(axis=0).A1
once = (counts == 1).sum()
print(f"\n{once:,} words appear once in all of training, among them")
print("  " + ", ".join(sorted(words[counts == 1])[100:106]))
