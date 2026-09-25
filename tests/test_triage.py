"""
Chapter 20's ticket triage, on tickets made up in the test: tokens and
the rubric, the measures, the trained models, the router and the
language-model client. Nothing here touches the network or needs the
warehouse; the language model is a stand-in object.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from foresight.triage import llm
from foresight.triage.evaluate import (desk_pairs, eval_set, kappa,
                                       paired, per_class,
                                       urgent_at_capacity)
from foresight.triage.features import (CATEGORIES, PRIORITIES,
                                       in_transit, relabel, split,
                                       tokens)
from foresight.triage.model import TriageModel
from foresight.triage.neural import (TransformerTriage, Vocabulary,
                                     average_vectors, word_vectors)
from foresight.triage.probes import probes
from foresight.triage.router import route, unsure

KINDS = [  # body, category, priority
    ("order {n} still hasnt arrived", "Delivery", "Normal"),
    ("{n} boxes missing from order {n}", "Delivery", "High"),
    ("invoice INV-{n} charged twice", "Billing", "Normal"),
    ("MRD-CLE-00{d} falling apart after one use", "Quality", "High"),
    ("we need MRD-SAF-01{d} today or the line stops", "Stock", "Urgent"),
    ("send me our current price list", "Account", "Low"),
]


def corpus(n: int = 120, seed: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        body, cat, pri = KINDS[i % len(KINDS)]
        rows.append({"body": body.format(n=g.integers(100, 999),
                                         d=g.integers(1, 9)),
                     "category": cat, "priority": pri})
    t = pd.DataFrame(rows)
    t["ticket_id"] = [f"T-{i:04d}" for i in range(n)]
    t["opened_at"] = pd.date_range("2024-01-01", periods=n, freq="6h")
    t["day"] = t.opened_at.dt.normalize()
    t["desk"] = np.where(np.arange(n) % 2, "North desk", "South desk")
    return t


# --------------------------------------------------------------- text
def test_tokens_replace_codes_and_numbers_with_what_they_are():
    body = "Hi, MRD-CLE-001 missing from order 1234 on INV-55555"
    assert tokens(body) == ["hi", "sku", "missing", "from", "order",
                            "num", "on", "inv"]
    assert tokens("Plus de MRD-SAF-012 en stock, c'est urgent.") == [
        "plus", "de", "sku", "en", "stock", "c", "est", "urgent"]


def test_the_rubric_moves_transit_damage_to_delivery_only():
    t = pd.DataFrame({
        "body": ["MRD-PAC-027 arrived crushed, outer box soaked",
                 "driver dropped the delivery, it split open",
                 "MRD-CLE-001 falling apart after one use"],
        "category": ["Quality", "Billing", "Quality"]})
    assert in_transit(t.body[0]) and not in_transit(t.body[2])
    assert relabel(t).tolist() == ["Delivery", "Delivery", "Quality"]


def test_split_is_by_the_day_opened():
    t = pd.DataFrame({"opened_at": pd.to_datetime(
        ["2024-06-30 23:59", "2024-07-01 00:00", "2025-01-01 08:00"])})
    train, valid, test = split(t)
    assert (len(train), len(valid), len(test)) == (1, 1, 1)


# ----------------------------------------------------------- measures
def test_per_class_precision_and_recall():
    table = per_class(["A", "A", "B", "B"], ["A", "B", "B", "B"],
                      ["A", "B"])
    assert table.loc["A", "precision"] == 1.0
    assert table.loc["A", "recall"] == 0.5
    assert table.loc["B", "precision"] == pytest.approx(2 / 3)


def test_kappa_is_one_for_agreement_and_zero_for_chance():
    assert kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    a = ["x", "x", "y", "y"]
    assert kappa(a, ["x", "y", "x", "y"]) == pytest.approx(0.0)


def test_urgent_at_capacity_reads_the_top_k_each_day():
    t = pd.DataFrame({
        "day": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 2),
        "opened_at": pd.date_range("2025-01-01", periods=5, freq="h"),
        "priority": ["Low", "Urgent", "Urgent", "Urgent", "Low"]})
    score = [0.9, 0.8, 0.1, 0.7, 0.2]
    assert urgent_at_capacity(t, score, k=1) == pytest.approx(1 / 3)
    assert urgent_at_capacity(t, score, k=2) == pytest.approx(2 / 3)


def test_paired_difference_and_interval():
    d, lo, hi = paired([1, 1, 1, 0] * 25, [1, 0, 1, 0] * 25, reps=200)
    assert d == pytest.approx(0.25) and lo <= d <= hi


def test_desk_pairs_matches_near_identical_tickets():
    t = pd.DataFrame({
        "body": ["box arrived crushed", "box arrived crushed",
                 "send me our price list", "send me our price list"],
        "desk": ["North desk", "South desk"] * 2,
        "category": ["Quality", "Delivery", "Account", "Account"]})
    pairs = desk_pairs(t)
    assert pairs[["north", "south"]].values.tolist() == [
        ["Quality", "Delivery"], ["Account", "Account"]]


def test_the_comparison_set_is_fixed():
    t = corpus(300)
    a, b = eval_set(t, n=50), eval_set(t, n=50)
    assert a.ticket_id.tolist() == b.ticket_id.tolist()
    assert a.opened_at.is_monotonic_increasing


# ------------------------------------------------------------- models
def test_tfidf_model_learns_and_repeats_itself():
    t = corpus()
    m = TriageModel().fit(t.body, t.priority, t.category)
    p = m.predict(t.body)
    assert (p.category == t.category).mean() == 1.0
    assert set(p.columns) >= {"priority", "category", "p_urgent",
                              "priority_conf", "category_conf"}
    q = TriageModel().fit(t.body, t.priority, t.category).predict(t.body)
    assert np.array_equal(p.p_urgent, q.p_urgent)


def test_vocabulary_pads_and_marks_unknown_words():
    v = Vocabulary([["a", "b", "a"], ["b", "c"]], min_count=2)
    ids = v.encode([["a", "z"]], length=4)
    assert ids.tolist() == [[v.index["a"], 1, 0, 0]]


def test_word_vectors_have_one_row_per_word(seed):
    lists = [tokens(b) for b in corpus(60).body]
    v = Vocabulary(lists)
    W = word_vectors(lists, v, dim=8, epochs=1)
    assert W.shape == (len(v), 8)
    assert average_vectors(lists[:3], v, W).shape == (3, 8)


def test_transformer_trains_and_repeats_itself(seed):
    t = corpus(90)
    fit = dict(bodies=t.body, priority=t.priority, category=t.category)
    a = TransformerTriage(epochs=3, dim=16).fit(**fit).predict(t.body)
    b = TransformerTriage(epochs=3, dim=16).fit(**fit).predict(t.body)
    assert np.array_equal(a.p_urgent, b.p_urgent)
    assert set(a.priority) <= set(PRIORITIES)
    assert set(a.category) <= set(CATEGORIES)


# ------------------------------------------------------------- router
class Fixed:
    """A model whose answers and confidences are set by hand."""

    def __init__(self, conf):
        self.conf = conf

    def predict(self, bodies):
        n = len(bodies)
        return pd.DataFrame({"priority": ["Normal"] * n,
                             "category": ["Delivery"] * n,
                             "p_urgent": [0.0] * n,
                             "priority_conf": self.conf,
                             "category_conf": [0.99] * n})


def test_only_unsure_tickets_are_sent_and_answered():
    model = Fixed([0.95, 0.4, 0.5])
    sent = []

    def ask(bodies):
        sent.extend(bodies)
        return pd.DataFrame({"category": ["Quality", None],
                             "priority": ["Urgent", None]})

    out = route(["a", "b", "c"], model, ask=ask, threshold=0.7)
    assert sent == ["b", "c"]
    assert out.answered_by.tolist() == ["model", "llm", "model (unsure)"]
    assert out.category.tolist() == ["Delivery", "Quality", "Delivery"]
    assert unsure(model.predict("abc"), 0.45).tolist() == [False, True,
                                                           False]


# ----------------------------------------------------- language model
def test_the_prompt_carries_the_rubric_and_the_ticket():
    m = llm.messages("pallet damaged in transit")
    assert "damaged in transit" in m[0]["content"]
    assert m[1]["content"].endswith("pallet damaged in transit")


def test_replies_are_parsed_or_refused():
    ok = llm.parse('{"category": "Stock", "priority": "Urgent"}')
    assert (ok.category, ok.priority) == ("Stock", "Urgent")
    assert llm.parse('{"category": "Stock", "priority": "Soon"}') is None
    assert llm.parse("I think it is a stock question") is None
    assert llm.parse(None) is None


def test_no_rates_means_no_run(monkeypatch):
    monkeypatch.delenv("RATE_TEST_MODEL_INPUT", raising=False)
    with pytest.raises(llm.OverBudget):
        llm.Budget(model="test-model")


class FakeAPI:
    """Stands in for the client: same call, canned replies, no network."""

    def __init__(self, content):
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create))
        self.content = content

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["response_format"] == llm.SCHEMA
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=400, completion_tokens=20))


def test_classify_counts_spend_and_stops_at_the_ceiling(monkeypatch):
    monkeypatch.setenv("RATE_TEST_MODEL_INPUT", "1000")
    monkeypatch.setenv("RATE_TEST_MODEL_OUTPUT", "1000")
    api = FakeAPI('{"category": "Billing", "priority": "Normal"}')
    budget = llm.Budget(ceiling=2.0, model="test-model")
    out = llm.classify(["a", "b"], api=api, budget=budget,
                       model="test-model")
    assert out.category.tolist() == ["Billing", "Billing"]
    assert budget.spent == pytest.approx(0.84)     # 2 x 420 tokens
    with pytest.raises(llm.OverBudget):
        llm.classify(["c", "d", "e"], api=api, budget=budget,
                     model="test-model")
    assert api.calls == 3


def test_the_probes_use_the_books_labels():
    p = probes()
    assert len(p) == 22
    assert set(p.category) <= set(CATEGORIES)
    assert set(p.priority) <= set(PRIORITIES)
