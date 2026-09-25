# A language model with a prompt, on the comparison set and the probes.
# skip
# awaits-key: runs once OPENAI_API_KEY is set (TODO-AUTHOR)
# nondeterministic: live model calls and wall-clock latency
# timeout: 1800
import json

from foresight.config import SPEND_CEILING_USD
from foresight.triage.evaluate import eval_set, per_class
from foresight.triage.features import (PRIORITIES, relabel, split,
                                       tickets)
from foresight.triage.llm import Budget, classify
from foresight.triage.probes import probes

t = tickets()
t["category"] = relabel(t)
_, _, test = split(t)
sample = eval_set(test)                 # the same 1,000 as every model
probe = probes()

budget = Budget()               # refuses to start without rates in .env
answers = classify(sample.body, budget=budget)
probed = classify(probe.body, budget=budget)

ok = answers.category.notna() & answers.priority.notna()
print(f"{len(answers):,} tickets; {ok.mean():.1%} of replies fit the "
      f"schema")
for label in ["priority", "category"]:
    acc = (answers[label] == sample[label]).mean()
    print(f"  {label} accuracy {acc:.1%}")
both = ((answers.priority == sample.priority)
        & (answers.category == sample.category))
print(f"  both right {both.mean():.1%}\n")
print(per_class(sample.priority, answers.priority, PRIORITIES)
      .to_string(float_format=lambda v: f"{v:.3f}"))

s = answers.seconds
calls = len(answers) + len(probed)
tokens_each = (answers.prompt_tokens + answers.completion_tokens).mean()
per_1000 = 1000 * budget.spent / calls
print(f"\nlatency: median {s.median():.2f}s, 90th percentile "
      f"{s.quantile(0.9):.2f}s, slowest {s.max():.2f}s")
print(f"tokens per ticket {tokens_each:.0f}; ${per_1000:.3f} per 1,000 "
      f"tickets")
print(f"spent ${budget.spent:.4f} of the ${SPEND_CEILING_USD:.2f} "
      f"ceiling")

right = ((probed.category == probe.category)
         & (probed.priority == probe.priority))
print("\nprobes, both labels right")
for kind, hit in right.groupby(probe.kind, sort=False):
    print(f"  {kind:15}{hit.sum()} of {len(hit)}")

with open("code/20/11_llm_classifier.json", "w") as f:
    json.dump({"eval": answers.to_dict(orient="records"),
               "probes": probed.to_dict(orient="records"),
               "per_1000_usd": per_1000,
               "median_s": float(s.median())}, f)
