# Learn two numbers for each product from the orders it appears in.
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
        "SELECT DISTINCT l.order_id, l.sku, o.order_date"
        " FROM order_lines l"
        " JOIN orders o USING (order_id)"
        " WHERE o.order_date < '2025-01-01'", con)
    products = pd.read_sql(
        "SELECT sku, category FROM products ORDER BY sku", con)
sku = products.sku.to_numpy()
index = pd.Series(range(len(sku)), index=sku)


def pair_counts(lines):
    """How often each pair of products shares an order, and how often
    it would if products were put into orders at random."""
    order = pd.factorize(lines.order_id)[0]
    basket = coo_matrix((np.ones(len(lines)),
                         (order, index[lines.sku].to_numpy()))).tocsr()
    together = (basket.T @ basket).toarray()
    np.fill_diagonal(together, 0)
    chance = np.outer(together.sum(1), together.sum(1)) / together.sum()
    np.fill_diagonal(chance, 0)
    return together, chance


together, chance = pair_counts(lines)
print(f"{lines.order_id.nunique():,} orders before 2025,"
      f" {len(sku)} products")

# One vector per product. Pairs bought together should score high,
# pairs expected by chance low: word2vec's objective, on baskets.
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


def neighbours(M, k=5):
    """Each product's k nearest others, by the angle between vectors."""
    U = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


cat = products.category.to_numpy()
near = neighbours(V)
print(f"\nNearest five to {sku[0]}, the Pemberton cloth ({cat[0]}):")
for j in near[0]:
    print(f"  {sku[j]}  {cat[j]}")
same = cat[near] == cat[:, None]
print("\nShare of the nearest five in the product's own category")
for c in sorted(set(cat)):
    print(f"  {c:<38}{same[cat == c].mean():>6.0%}")
print(f"  {'all 48 products':<38}{same.mean():>6.0%}")
pairs = (cat[:, None] == cat[None, :]).sum() - len(cat)
print(f"  {'by chance':<38}{pairs / (len(cat) * 47):>6.0%}")
counts = cat[neighbours(together.astype(float))] == cat[:, None]
print(f"  {'raw pair counts, 48 numbers each':<38}"
      f"{counts.mean():>6.0%}")

san = np.outer(cat == "Sanitation", cat == "Sanitation")
print("\nSanitation pairs in one order, as a multiple of chance")
since = lines[lines.order_date >= "2024-04-01"]
for name, rows in (("all orders before 2025", lines),
                   ("orders since its launch", since)):
    t, c = pair_counts(rows)
    print(f"  {name:<38}{t[san].sum() / c[san].sum():>6.2f}")

