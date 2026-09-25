# Exercise 4: product vectors from orders since Sanitation launched.
import sqlite3

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch.nn.functional import logsigmoid

from foresight.config import ML_WAREHOUSE, SEED, seed_everything

seed_everything()
with sqlite3.connect(ML_WAREHOUSE) as con:
    lines = pd.read_sql(
        "SELECT DISTINCT l.order_id, l.sku FROM order_lines l"
        " JOIN orders o USING (order_id)"
        " WHERE o.order_date >= '2024-04-01'"
        " AND o.order_date < '2025-01-01'", con)
    products = pd.read_sql(
        "SELECT sku, category FROM products ORDER BY sku", con)
sku, cat = products.sku.to_numpy(), products.category.to_numpy()
index = pd.Series(range(len(sku)), index=sku)
order = pd.factorize(lines.order_id)[0]
basket = coo_matrix((np.ones(len(lines)),
                     (order, index[lines.sku].to_numpy()))).tocsr()
together = (basket.T @ basket).toarray()
np.fill_diagonal(together, 0)
chance = np.outer(together.sum(1), together.sum(1)) / together.sum()
np.fill_diagonal(chance, 0)

torch.manual_seed(SEED)
vectors = torch.nn.Embedding(len(sku), 2)
opt = torch.optim.Adam(vectors.parameters(), lr=0.05)
pos = torch.tensor(together, dtype=torch.float32)
neg = torch.tensor(chance, dtype=torch.float32)
for step in range(1_000):
    opt.zero_grad()
    score = vectors.weight @ vectors.weight.T
    loss = -(pos * logsigmoid(score) + neg * logsigmoid(-score)).sum()
    (loss / pos.sum()).backward()
    opt.step()
V = vectors.weight.detach().numpy()
U = V / np.linalg.norm(V, axis=1, keepdims=True)
S = U @ U.T
np.fill_diagonal(S, -np.inf)
same = cat[np.argsort(-S, axis=1, kind="stable")[:, :5]] == cat[:, None]
print(f"{lines.order_id.nunique():,} orders, April to December 2024")
print("Share of the nearest five in the product's own category")
for c in sorted(set(cat)):
    print(f"  {c:<20}{same[cat == c].mean():>6.0%}")
print(f"  {'all 48 products':<20}{same.mean():>6.0%}")
pairs = (cat[:, None] == cat[None, :]).sum() - len(cat)
print(f"  {'by chance':<20}{pairs / (len(cat) * 47):>6.0%}")
