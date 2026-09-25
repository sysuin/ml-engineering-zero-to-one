# TF-IDF and logistic regression: the baseline every model must beat.
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from foresight.triage.evaluate import (CAPACITY, keyword_rule,
                                       per_class, urgent_at_capacity)
from foresight.triage.features import (CATEGORIES, PRIORITIES, split,
                                       tickets, tokens)

train, valid, _ = split(tickets())
vec = TfidfVectorizer(tokenizer=tokens, lowercase=False,
                      token_pattern=None, ngram_range=(1, 2), min_df=2,
                      sublinear_tf=True)
X, Xv = vec.fit_transform(train.body), vec.transform(valid.body)
models, preds = {}, {}
for label in ["priority", "category"]:
    models[label] = LogisticRegression(max_iter=2000)
    models[label].fit(X, train[label])
    preds[label] = models[label].predict(Xv)

print(f"{X.shape[1]:,} columns: words and pairs seen twice or more\n")
result = {}
for label, classes in [("priority", PRIORITIES),
                       ("category", CATEGORIES)]:
    table = per_class(valid[label], preds[label], classes)
    acc = (preds[label] == valid[label]).mean()
    commonest = (valid[label] == train[label].mode()[0]).mean()
    print(f"{label}: accuracy {acc:.1%} (always the commonest "
          f"class: {commonest:.1%})")
    print(table.to_string(float_format=lambda v: f"{v:.3f}") + "\n")
    result[label] = {"accuracy": acc, "commonest": commonest}

p_urgent = models["priority"].predict_proba(Xv)[
    :, list(models["priority"].classes_).index("Urgent")]
print(f"Urgent found in the first {CAPACITY} read each day")
scores = {"arrival": -valid.opened_at.astype("int64"),
          "urgent words": valid.body.map(keyword_rule),
          "TF-IDF model": p_urgent}
for name, s in scores.items():
    result[name] = urgent_at_capacity(valid, s)
    print(f"  {name:14}{result[name]:.1%}")

w = models["priority"].coef_[list(models["priority"].classes_)
                             .index("Urgent")]
top = np.argsort(-w)[:6]
names = vec.get_feature_names_out()
print("\nlargest weights towards Urgent: "
      + ", ".join(f"'{names[j]}'" for j in top[:3]))
print("  " + ", ".join(f"'{names[j]}'" for j in top[3:]))
with open("code/20/03_tfidf_baseline.json", "w") as f:
    json.dump(result, f, indent=1)
