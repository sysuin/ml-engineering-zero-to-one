# Exercise 1: three calls from October's list, with the facts behind
# their reasons, to be written up in an account manager's words.
from foresight.score import score

result = score("2024-10-02", checks=False)
listed = result["list"].set_index("rank")
for rank in (2, 20, 35):
    r = listed.loc[rank]
    print(f"{rank:>2}. {r.contract_id} {r['name']}, {r.segment},"
          f" {r.region}: {r.chance:.1%}{' (below)' if r.below else ''}")
    print(f"    last order {r.days_since_order} days ago; orders"
          f" {r.orders_90d} this quarter, {r.orders_prev_90d} before")
    print(f"    spend ${r.spend_365:,.0f}; discount {r.discount_pct}%;"
          f" customer {r.tenure_days / 365.25:.1f} years")
    for why in r.reasons:
        print(f"    - {why}")
