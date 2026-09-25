"""
Ticket triage by a hosted language model: the rubric as a prompt, the
answer as structured output, and a spending rail.

    MODEL_TRIAGE          the one place the model is named
    messages(body)        the prompt for one ticket
    parse(text)           the model's reply as a Triage, or None
    Budget                stops before SPEND_CEILING_USD is passed
    classify(bodies)      every ticket, with tokens and seconds each

The key is read from the environment (config.py loads .env), never
from code. Rates in dollars per million tokens come from the same
place, as RATE_<MODEL>_INPUT and RATE_<MODEL>_OUTPUT, so no price is
written here: Appendix C lists them.
"""
from __future__ import annotations

import os
import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ValidationError

from foresight.config import SPEND_CEILING_USD
from foresight.triage.features import CATEGORIES, PRIORITIES

MODEL_TRIAGE = os.getenv("FORESIGHT_MODEL_TRIAGE", "gpt-5.4-mini")

RUBRIC = """\
You triage support tickets for Meridian Supply Co., a wholesaler of
cleaning, safety, packaging and facilities supplies. Tickets may be in
any language. Give each ticket one category and one priority.

Category
- Delivery: an order that is late or has items missing, or goods that
  arrived damaged in transit (crushed, wet, dropped, torn packaging).
- Quality: a product that is faulty, inconsistent or unsafe in use.
- Billing: invoices, charges, credit notes, tax.
- Returns: sending goods back, return labels, collections.
- Account: account details, contacts, addresses, price lists,
  statements, or ending the contract.
- Stock: whether a product is available.

Priority
- Urgent: a risk to someone's safety, or goods missing or out of stock
  that stop the customer's site or line now.
- High: a faulty product; items missing from an order; a late order or
  a stock question where the customer says work is affected.
- Normal: late orders, damage in transit, double charges, returns,
  stock questions, ending the contract.
- Low: account changes, credit notes, tax and invoice paperwork."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triage", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "priority": {"type": "string", "enum": PRIORITIES},
            },
            "required": ["category", "priority"],
            "additionalProperties": False,
        },
    },
}


class Triage(BaseModel):
    category: Literal["Account", "Billing", "Delivery", "Quality",
                      "Returns", "Stock"]
    priority: Literal["Urgent", "High", "Normal", "Low"]


def messages(body: str) -> list[dict]:
    return [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Ticket:\n{body}"}]


def parse(text: str | None) -> Triage | None:
    """The reply as a Triage, or None if it does not fit the schema."""
    try:
        return Triage.model_validate_json(text or "")
    except ValidationError:
        return None


# ------------------------------------------------------------ spending
def rates(model: str = MODEL_TRIAGE) -> tuple[float, float] | None:
    """Dollars per million tokens (input, output), if set in .env."""
    key = model.upper().replace("-", "_").replace(".", "_")
    raw_in = os.getenv(f"RATE_{key}_INPUT")
    raw_out = os.getenv(f"RATE_{key}_OUTPUT")
    if raw_in and raw_out:
        return float(raw_in), float(raw_out)
    return None


class OverBudget(RuntimeError):
    pass


class Budget:
    """Counts dollars as replies arrive and refuses the next call once
    the ceiling would be passed. Without rates it cannot count dollars,
    so it refuses to start: an unpriced run is an unbounded one."""

    def __init__(self, ceiling: float = SPEND_CEILING_USD,
                 model: str = MODEL_TRIAGE):
        self.ceiling, self.rates = ceiling, rates(model)
        self.spent = 0.0
        if self.rates is None:
            raise OverBudget(f"no RATE_ values for {model} in .env")

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        r_in, r_out = self.rates
        return (prompt_tokens * r_in + completion_tokens * r_out) / 1e6

    def charge(self, prompt_tokens: int, completion_tokens: int):
        self.spent += self.cost(prompt_tokens, completion_tokens)

    def check(self, next_call: float) -> None:
        if self.spent + next_call > self.ceiling:
            raise OverBudget(f"${self.spent:.4f} spent; the next call "
                             f"would pass ${self.ceiling:.2f}")


def client():
    """An API client, with the key from the environment."""
    from openai import OpenAI       # optional: only Chapter 20 needs it
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (see .env)")
    return OpenAI()


def classify(bodies, api=None, budget: Budget | None = None,
             model: str = MODEL_TRIAGE) -> pd.DataFrame:
    """Ask the model about each ticket in turn. Returns its category
    and priority (None where the reply did not parse), the tokens each
    call used and the seconds each took."""
    api = api or client()
    budget = budget or Budget(model=model)
    rows, last = [], 0.0
    for body in bodies:
        budget.check(2 * last)          # room for a call twice the last
        started = time.perf_counter()
        reply = api.chat.completions.create(
            model=model, messages=messages(body),
            response_format=SCHEMA, temperature=0,
            max_completion_tokens=300)
        seconds = time.perf_counter() - started
        usage = reply.usage
        budget.charge(usage.prompt_tokens, usage.completion_tokens)
        last = budget.cost(usage.prompt_tokens, usage.completion_tokens)
        answer = parse(reply.choices[0].message.content)
        rows.append({
            "category": answer.category if answer else None,
            "priority": answer.priority if answer else None,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "seconds": seconds})
    out = pd.DataFrame(rows)
    out.attrs["spent"] = budget.spent
    return out

