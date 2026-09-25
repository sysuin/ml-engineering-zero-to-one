# Exercise 2: the break-even volumes, as formulas, under assumptions.
# Monthly cost at V tickets a month, with upkeep F dollars a month
# for the trained model, c dollars per 1,000 tickets for the LLM, and
# a share r of tickets sent on by the router:
#   LLM alone    V * c / 1000
#   trained      F
#   hybrid       F + r * V * c / 1000
F = 6 * 90          # assumed: six hours a month at $90 an hour
r = 0.07            # the router's share on the 2025 comparison set
print(f"upkeep ${F} a month, router sends {r:.0%}")
print(f"{'$ per 1,000':>12}{'trained < LLM':>16}{'hybrid < LLM':>15}")
for c in [0.5, 2, 10, 50]:
    print(f"{c:12.2f}{F / (c / 1000):16,.0f}"
          f"{F / ((1 - r) * c / 1000):15,.0f}")
print("hybrid < trained alone: never, on cost; it is bought for the")
print("tickets the model is unsure of, not to save money")
