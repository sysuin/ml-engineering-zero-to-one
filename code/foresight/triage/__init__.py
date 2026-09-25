"""Foresight's ticket triage: a priority and a category for every
support ticket, from its text, with the unsure ones sent on to a
language model (Chapter 20).

    features.py   the tickets, the time split, tokens, the rubric
    model.py      TF-IDF and logistic regression: the shipped model
    neural.py     word vectors and a small transformer, from scratch
    evaluate.py   per class, at capacity, agreement, the comparison set
    probes.py     hand-written tickets unlike the training years
    llm.py        the prompted model, its schema and its spending rail
    router.py     the hybrid: the model first, the unsure sent on
    report.py     python -m foresight.triage.report [--llm]
"""
