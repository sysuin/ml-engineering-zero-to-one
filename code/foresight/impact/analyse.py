"""
Reading a retention test: did the called accounts leave less often
than the held-out ones? Chapter 25 writes it.

Everything here works from what Meridian records and nothing else:
one row per listed contract with its cohort (`moment`), its `arm`
("called" or "held out", from holdout.py) and whether it `left`. It
never sees a true chance, a save or a simulated world.

    compare(records)       the difference between the arms, with a
                           bootstrap interval, a re-randomisation
                           p-value, and what it means in saves and
                           dollars
    sample_size(p0, p1)    contracts per arm for a given power
    power(p0, p1, n)       the chance a test of n per arm sees p0 - p1
    page(result)           the result as text no wider than 68
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from foresight.config import SEED
from foresight.costs import CostMatrix
from foresight.decide import COSTS

REPS = 2000
CALLED, HELD = "called", "held out"


def rates(records: pd.DataFrame) -> pd.DataFrame:
    """Contracts, leavers and the share that left, by arm."""
    g = records.groupby("arm").left
    return pd.DataFrame({"n": g.size(), "left": g.sum(),
                         "rate": g.mean()})


def _diff(left, is_held) -> float:
    return left[is_held].mean() - left[~is_held].mean()


def _cells(records: pd.DataFrame):
    """Row positions of each (cohort, arm) cell, for resampling."""
    key = records.moment.astype(str) + "|" + records.arm.astype(str)
    return [np.flatnonzero(key.to_numpy() == k) for k in key.unique()]


def bootstrap(records: pd.DataFrame, reps: int = REPS,
              seed: int = SEED) -> np.ndarray:
    """The held-out minus called difference, recomputed on `reps`
    resamples of contracts within each cohort and arm."""
    left = records.left.to_numpy(float)
    held = records.arm.to_numpy() == HELD
    cells = _cells(records)
    g = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([g.choice(c, len(c)) for c in cells])
        out[r] = _diff(left[idx], held[idx])
    return out


def rerandomise(records: pd.DataFrame, reps: int = REPS,
                seed: int = SEED) -> np.ndarray:
    """The difference under `reps` fresh draws of the holdout, made
    the way holdout.py made the real one (within each cohort). If calls
    did nothing, the real difference is one more draw like these."""
    left = records.left.to_numpy(float)
    held = records.arm.to_numpy() == HELD
    cohorts = [np.flatnonzero(records.moment.to_numpy() == m)
               for m in records.moment.unique()]
    g = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        shuffled = held.copy()
        for c in cohorts:
            shuffled[c] = g.permutation(held[c])
        out[r] = _diff(left, shuffled)
    return out


def compare(records: pd.DataFrame, costs: CostMatrix = COSTS,
            reps: int = REPS, seed: int = SEED) -> dict:
    """The test's result. `diff` is how many points less often the
    called arm left; `save_rate` is that as a share of the held-out
    arm's leavers, the number the brief assumed."""
    r = rates(records)
    held, called = r.loc[HELD, "rate"], r.loc[CALLED, "rate"]
    diff = held - called
    boot = bootstrap(records, reps, seed)
    null = rerandomise(records, reps, seed)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    per_call = [d * costs.value_at_stake - costs.call_cost
                for d in (diff, lo, hi)]
    return {"rates": r, "diff": diff, "lo": lo, "hi": hi,
            "p_value": float(np.mean(np.abs(null)
                                     >= abs(diff) - 1e-12)),
            "save_rate": diff / held, "save_lo": lo / held,
            "save_hi": hi / held, "saves": diff * r.loc[CALLED, "n"],
            "per_call": per_call, "cohorts": records.moment.nunique()}


def before_after(before: pd.DataFrame, after: pd.DataFrame) -> float:
    """The comparison people make first: the share of the list that
    left before the calls began, minus the share after. Positive reads
    as 'calls kept accounts', like compare()'s difference. Both frames
    carry `left`."""
    return before.left.mean() - after.left.mean()


# ------------------------------------------------ design
def sample_size(p0: float, p1: float, alpha: float = 0.05,
                power: float = 0.80) -> float:
    """Contracts per arm for a two-sided test at `alpha` to see a drop
    from p0 to p1 with the given power (Appendix F)."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    spread = p0 * (1 - p0) + p1 * (1 - p1)
    return (za + zb) ** 2 * spread / (p0 - p1) ** 2


def power(p0: float, p1: float, n: float,
          alpha: float = 0.05) -> float:
    """The chance a test with n per arm finds a drop from p0 to p1."""
    if p0 == p1:
        return alpha
    spread = p0 * (1 - p0) + p1 * (1 - p1)
    z = abs(p0 - p1) / np.sqrt(spread / n)
    return float(norm.cdf(z - norm.ppf(1 - alpha / 2)))


def cohorts_needed(p0: float, p1: float, held: int, called: int,
                   alpha: float = 0.05, power: float = 0.80) -> float:
    """Monthly cohorts a test needs when each holds out `held` of its
    list and calls `called`: the same arithmetic, arms of any size."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    per_cohort = p0 * (1 - p0) / held + p1 * (1 - p1) / called
    return (za + zb) ** 2 * per_cohort / (p0 - p1) ** 2


def simulated_power(outcomes: np.ndarray, save: float, cohorts: int,
                    held: int = 20, reps: int = REPS, seed: int = SEED,
                    alpha: float = 0.05) -> float:
    """Power by simulation, from lists already on record. `outcomes`
    holds one past cohort's list per row (1 for left), with nobody
    called; they are reused in turn for as many cohorts as the test
    runs. In each of `reps` tests a cohort's list is its row resampled
    with replacement, `held` of it are held out at random, and a call
    keeps a would-be leaver with chance `save`. Returns the share of
    tests whose two-proportion test (Appendix F) finds a difference
    at `alpha`."""
    g = np.random.default_rng(seed)
    base = np.asarray(outcomes, dtype=bool)
    rows = np.arange(cohorts) % len(base)
    k = base.shape[1]
    n0, n1 = held * cohorts, (k - held) * cohorts
    za = norm.ppf(1 - alpha / 2)
    found = 0
    for _ in range(reps):
        left = base[rows[:, None], g.integers(0, k, (cohorts, k))]
        held_out = g.random(left.shape).argsort(axis=1) < held
        left &= ~(~held_out & (g.random(left.shape) < save))
        r0, r1 = left[held_out].mean(), left[~held_out].mean()
        pooled = left.mean()
        se = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
        found += bool(se > 0 and abs(r0 - r1) / se > za)
    return found / reps


# ------------------------------------------------ the page
def dollars(v: float) -> str:
    """-$19 rather than $-19."""
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def p_text(p: float, reps: int = REPS) -> str:
    """A re-randomised p-value, never printed as zero."""
    return f"<{1 / reps:.4f}" if p == 0 else f"{p:.3f}"


def page(result: dict, title: str) -> str:
    r = result["rates"]
    lines = [title, ""]
    for arm in (HELD, CALLED):
        lines.append(f"  {arm:<10}{r.loc[arm, 'n']:>5} contracts"
                     f"{r.loc[arm, 'left']:>5} left"
                     f"{r.loc[arm, 'rate']:>8.1%}")
    d, lo, hi = (100 * result[k] for k in ("diff", "lo", "hi"))
    s, slo, shi = (100 * result[k]
                   for k in ("save_rate", "save_lo", "save_hi"))
    v, vlo, vhi = result["per_call"]
    lines += ["",
              f"  held out minus called {d:+5.1f} points"
              f" ({lo:+.1f} to {hi:+.1f})",
              f"  p-value, re-randomised {p_text(result['p_value'])}",
              f"  save rate {s:5.1f}% of would-be leavers"
              f" ({slo:.1f}% to {shi:.1f}%)",
              f"  net per call {dollars(v)} ({dollars(vlo)} to"
              f" {dollars(vhi)})"]
    return "\n".join(lines)
