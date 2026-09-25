"""
Text models learned from Meridian's own tickets, with nothing
downloaded: word vectors trained on the ticket corpus, and a small
transformer trained from scratch on a CPU.

    Vocabulary(token_lists)      words seen at least twice, as numbers
    word_vectors(token_lists)    skip-gram with negative sampling
    average_vectors(...)         a ticket as its words' mean vector
    TicketTransformer            embeddings, one attention layer, and
                                 two heads: priority and category
    TransformerTriage            fit / predict, like TriageModel
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from foresight.config import SEED
from foresight.triage.features import CATEGORIES, PRIORITIES, tokens

PAD, UNKNOWN = 0, 1


class Vocabulary:
    """Every word seen at least min_count times gets a number; the rest
    share UNKNOWN. Number 0 is padding."""

    def __init__(self, token_lists, min_count: int = 2):
        counts = Counter(w for ts in token_lists for w in ts)
        kept = sorted(w for w, n in counts.items() if n >= min_count)
        self.words = ["<pad>", "<unk>"] + kept
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = np.array([0, 0] + [counts[w] for w in kept])

    def __len__(self) -> int:
        return len(self.words)

    def ids(self, ts) -> list[int]:
        return [self.index.get(w, UNKNOWN) for w in ts]

    def encode(self, token_lists, length: int = 32) -> torch.Tensor:
        """A row of word numbers per ticket, cut or padded to length."""
        out = torch.zeros(len(token_lists), length, dtype=torch.long)
        for row, ts in enumerate(token_lists):
            ids = self.ids(ts)[:length]
            out[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out


# ------------------------------------------------------- word vectors
def skipgram_pairs(token_lists, vocab: Vocabulary, window: int = 2):
    """Every (word, neighbour) pair within `window` words."""
    centre, context = [], []
    for ts in token_lists:
        ids = vocab.ids(ts)
        for i, w in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    centre.append(w)
                    context.append(ids[j])
    return torch.tensor(centre), torch.tensor(context)


def word_vectors(token_lists, vocab: Vocabulary, dim: int = 32,
                 epochs: int = 3, negatives: int = 5, batch: int = 512,
                 lr: float = 0.01, seed: int = SEED) -> np.ndarray:
    """Train one vector per word so that words and their neighbours
    score high and randomly drawn words score low (word2vec's
    skip-gram with negative sampling). No labels are used."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    centre, context = skipgram_pairs(token_lists, vocab)
    noise = torch.tensor(vocab.counts, dtype=torch.float) ** 0.75
    inner = nn.Embedding(len(vocab), dim)
    outer = nn.Embedding(len(vocab), dim)
    nn.init.uniform_(inner.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(outer.weight)
    opt = torch.optim.Adam([*inner.parameters(), *outer.parameters()],
                           lr=lr)
    logsig = nn.functional.logsigmoid
    for _ in range(epochs):
        order = torch.randperm(len(centre), generator=g)
        for idx in order.split(batch):
            c, o = inner(centre[idx]), outer(context[idx])
            drawn = torch.multinomial(noise, len(idx) * negatives,
                                      replacement=True, generator=g)
            n = outer(drawn).view(len(idx), negatives, dim)
            real = logsig((c * o).sum(1))
            fake = logsig(-(n @ c.unsqueeze(2)).squeeze(2)).sum(1)
            loss = -(real + fake).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return inner.weight.detach().numpy().copy()


def average_vectors(token_lists, vocab: Vocabulary,
                    W: np.ndarray) -> np.ndarray:
    """Each ticket as the mean of its known words' vectors."""
    out = np.zeros((len(token_lists), W.shape[1]))
    for row, ts in enumerate(token_lists):
        ids = [i for i in vocab.ids(ts) if i != UNKNOWN]
        if ids:
            out[row] = W[ids].mean(axis=0)
    return out


def nearest(word: str, vocab: Vocabulary, W: np.ndarray,
            k: int = 4) -> list[str]:
    """The k words whose vectors point most nearly the same way."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    unit = W / np.maximum(norms, 1e-9)
    sims = unit @ unit[vocab.index[word]]
    order = [i for i in np.argsort(-sims) if i > UNKNOWN]
    return [vocab.words[i] for i in order if vocab.words[i] != word][:k]


# -------------------------------------------------------- transformer
class TicketTransformer(nn.Module):
    """Word and position embeddings, one layer of self-attention, the
    mean over the ticket's words, and a head for each label."""

    def __init__(self, words: int, dim: int = 32, heads: int = 2,
                 length: int = 32, dropout: float = 0.1):
        super().__init__()
        self.words = nn.Embedding(words, dim, padding_idx=PAD)
        self.places = nn.Embedding(length, dim)
        self.layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=2 * dim, dropout=dropout,
            batch_first=True)
        self.priority = nn.Linear(dim, len(PRIORITIES))
        self.category = nn.Linear(dim, len(CATEGORIES))

    def forward(self, ids: torch.Tensor):
        pad = ids == PAD
        places = torch.arange(ids.shape[1]).expand_as(ids)
        h = self.layer(self.words(ids) + self.places(places),
                       src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(2).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.priority(pooled), self.category(pooled)


class TransformerTriage:
    """The transformer behind TriageModel's interface."""

    def __init__(self, dim: int = 32, heads: int = 2, epochs: int = 12,
                 batch: int = 64, lr: float = 2e-3, seed: int = SEED):
        self.dim, self.heads, self.epochs = dim, heads, epochs
        self.batch, self.lr, self.seed = batch, lr, seed

    def fit(self, bodies, priority, category, watch=None):
        """Train for self.epochs passes. If watch is (bodies, priority,
        category), record both accuracies on it after every epoch."""
        ts = [tokens(b) for b in bodies]
        self.vocab_ = Vocabulary(ts)
        X = self.vocab_.encode(ts)
        yp = torch.tensor([PRIORITIES.index(p) for p in priority])
        yc = torch.tensor([CATEGORIES.index(c) for c in category])
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.net_ = TicketTransformer(len(self.vocab_), self.dim,
                                      self.heads)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.losses_, self.watched_ = [], []
        for _ in range(self.epochs):
            self.net_.train()
            total = 0.0
            order = torch.randperm(len(X), generator=g)
            for idx in order.split(self.batch):
                zp, zc = self.net_(X[idx])
                loss = loss_fn(zp, yp[idx]) + loss_fn(zc, yc[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.losses_.append(total / len(X))
            if watch is not None:
                p = self.predict(watch[0])
                self.watched_.append(
                    ((p.priority == np.asarray(watch[1])).mean(),
                     (p.category == np.asarray(watch[2])).mean()))
        return self

    def proba(self, bodies, label: str) -> pd.DataFrame:
        X = self.vocab_.encode([tokens(b) for b in bodies])
        self.net_.eval()
        with torch.no_grad():
            zp, zc = self.net_(X)
        z, names = ((zp, PRIORITIES) if label == "priority"
                    else (zc, CATEGORIES))
        return pd.DataFrame(torch.softmax(z, 1).numpy(), columns=names)

    def predict(self, bodies) -> pd.DataFrame:
        out = {}
        for label in ("priority", "category"):
            P = self.proba(bodies, label)
            out[label] = P.columns[np.argmax(P.to_numpy(), axis=1)]
            out[f"{label}_conf"] = P.max(axis=1).to_numpy()
            if label == "priority":
                out["p_urgent"] = P["Urgent"].to_numpy()
        return pd.DataFrame(out)
