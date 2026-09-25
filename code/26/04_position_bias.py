# Clicks depend on position: learn from them naively, and you learn
# the old ranker. Weight by how often each position is looked at.
import numpy as np

from foresight.config import rng

g = rng()
Q, K, IMPS = 3_000, 10, 200             # queries, results, impressions
rel = g.beta(1, 3, size=(Q, K))         # true chance of a click if seen
look = 1 / np.arange(1, K + 1) ** 0.9   # chance position k is looked at

# The ranker in production: right on average, noisy per query.
order = np.argsort(-(rel + g.normal(0, 0.25, (Q, K))), axis=1)
shown = np.take_along_axis(rel, order, axis=1)     # by position
clicks = g.binomial(IMPS, look * shown)


def ndcg(gains_by_position):
    disc = 1 / np.log2(np.arange(2, K + 2))
    ideal = np.sort(rel, axis=1)[:, ::-1]
    return float(((gains_by_position * disc).sum(1)
                  / (ideal * disc).sum(1)).mean())


def rerank(score):
    """Put each query's results in order of score; return true gains."""
    return np.take_along_axis(shown, np.argsort(-score, axis=1), axis=1)


# A small randomised experiment: 2% of impressions swap the top result
# into position k. Its click rate there, over its rate at the top,
# estimates how much less position k is looked at.
n_exp = int(0.02 * Q * IMPS)
q, k = g.integers(Q, size=n_exp), g.integers(1, K, size=n_exp)
hit = g.random(n_exp) < look[k] * shown[q, 0]
top_ctr = clicks[:, 0].sum() / (Q * IMPS)
est = np.ones(K)
for pos in range(1, K):
    est[pos] = hit[k == pos].mean() / top_ctr

print(f"{'position':>9}" + "".join(f"{p:>6}" for p in (1, 2, 3, 5, 10)))
print(f"{'true':>9}" + "".join(f"{look[p - 1]:>6.2f}"
                               for p in (1, 2, 3, 5, 10)))
print(f"{'estimated':>9}" + "".join(f"{est[p - 1]:>6.2f}"
                                    for p in (1, 2, 3, 5, 10)))
print(f"from {n_exp:,} swapped impressions of {Q * IMPS:,}\n")

# The pairs production got wrong: a better result shown lower down.
i, j = np.triu_indices(K, 1)                  # i above j on the page
wrong = shown[:, j] > shown[:, i] + 0.1       # j clearly better
far = wrong & (j - i >= 4)                    # four or more places


def fixed(score, pairs):
    """Share of these pairs that this score puts the right way."""
    return float((score[:, j] > score[:, i])[pairs].mean())


ctr = clicks / IMPS
page = np.tile(-np.arange(K), (Q, 1))          # the order as shown
print(f"{'':30}{'NDCG@10':>8}{'fixes':>7}{'far':>6}")
for name, score in (("production ranker", page),
                    ("raw click rate", ctr),
                    ("click rate / estimated look", ctr / est),
                    ("click rate / true look", ctr / look),
                    ("true relevance", shown)):
    print(f"  {name:28}{ndcg(rerank(score)):>8.3f}"
          f"{fixed(score, wrong):>7.0%}{fixed(score, far):>6.0%}")
print(f"{wrong.sum():,} wrong pairs, {far.sum():,} far apart")
