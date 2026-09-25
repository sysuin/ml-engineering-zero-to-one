# One validation contract's path through the depth-3 tree, in words.
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.models.logistic import TRAIN, VALIDATION, features, load

train, valid = load(*TRAIN), load(*VALIDATION)
X, Xv = features(train), features(valid)
tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
tree.fit(X, train.not_renewed)
p = tree.predict_proba(Xv)[:, 1]
top = np.lexsort((valid.contract_id, -p))[0]   # riskiest, lowest id
row = Xv.iloc[top]
print(f"Contract {valid.contract_id.iloc[top]}, account"
      f" {valid.account_id.iloc[top]}, marked"
      f" {valid.moment.iloc[top]:%d %B %Y}")
t = tree.tree_
node = 0
while t.children_left[node] >= 0:
    col, cut = X.columns[t.feature[node]], t.threshold[node]
    goes = "left" if row[col] <= cut else "right"
    print(f"  {col} is {row[col]:g}: {'<=' if goes == 'left' else '>'}"
          f" {cut:g}, so {goes}")
    step = t.children_left if goes == "left" else t.children_right
    node = step[node]
rows = int(t.n_node_samples[node])
print(f"  leaf: {rows} training contracts,"
      f" {t.value[node][0][1]:.1%} left")
same = p == p[top]
went = "left" if valid.not_renewed.iloc[top] else "renewed"
print(f"Validation contracts in the same leaf: {same.sum()},"
      f" of whom {valid.not_renewed[same].sum()} left; this one {went}")
