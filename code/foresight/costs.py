"""
What a call-or-not decision is worth, in dollars and account-manager hours.

Chapter 3 writes it for the renewal decision: the account team calls a
contract's account before the renewal conversation, or it does not, and the
contract then renews or it does not. Four outcomes, each with a price.
Chapter 7 prices a confusion matrix with it, Chapter 14 chooses a threshold
with it, and Chapter 25 replaces the assumed save rate with a measured one.

Every figure in a CostMatrix is an input somebody has to own. The value at
stake is measured from the warehouse; the save rate, the length of a call
and the price of an hour are estimates until Chapter 25, and the brief says
so.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostMatrix:
    value_at_stake: float  # a renewal is worth a year of profit, in $
    save_rate: float       # share of would-be leavers a call keeps
    call_hours: float      # account-manager hours a call takes, all in
    hour_cost: float       # dollars an account-manager hour costs

    @property
    def call_cost(self) -> float:
        """Dollars one call costs, whoever it goes to."""
        return self.call_hours * self.hour_cost

    @property
    def save_value(self) -> float:
        """Expected dollars one call to a would-be leaver recovers."""
        return self.save_rate * self.value_at_stake

    def cells(self) -> dict[str, float]:
        """
        Net dollars of each outcome, measured against making no call.

        tp  called, and would have left: expected save, less the call
        fp  called, and would have stayed: the call, wasted
        fn  not called, and left: nothing spent, nothing saved
        tn  not called, and stayed: nothing happens
        """
        return {"tp": self.save_value - self.call_cost,
                "fp": -self.call_cost, "fn": 0.0, "tn": 0.0}

    def error_costs(self) -> dict[str, float]:
        """What each kind of mistake costs, against getting it right."""
        return {"fp": self.call_cost,
                "fn": self.save_value - self.call_cost}

    def net_value(self, tp: float, fp: float,
                  fn: float = 0, tn: float = 0) -> float:
        """Net dollars of a set of decisions, from its four counts."""
        c = self.cells()
        return tp * c["tp"] + fp * c["fp"] + fn * c["fn"] + tn * c["tn"]

    def value_per_call(self, precision: float) -> float:
        """Net dollars a call, for a list this often on a leaver."""
        return precision * self.save_value - self.call_cost

    def break_even(self) -> float:
        """The chance of leaving above which a call pays for itself."""
        return self.call_cost / self.save_value
