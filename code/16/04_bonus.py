# A team of three splits a bonus by Shapley's rule: each person's share
# is what they add, averaged over every order in which they could join.
from itertools import permutations

team = ["manager", "analyst", "buyer"]
# What each group would have won on its own, in thousands of dollars.
won = {(): 0, ("manager",): 12, ("analyst",): 0, ("buyer",): 6,
       ("analyst", "manager"): 24, ("buyer", "manager"): 18,
       ("analyst", "buyer"): 6, ("analyst", "buyer", "manager"): 36}


def value(group):
    return won[tuple(sorted(group))]


share = dict.fromkeys(team, 0.0)
orders = list(permutations(team))
print(f"{'order of joining':<30}" + "".join(f"{p:>9}" for p in team))
for order in orders:
    added, so_far = {}, []
    for person in order:
        added[person] = value(so_far + [person]) - value(so_far)
        so_far.append(person)
        share[person] += added[person] / len(orders)
    print(f"{' > '.join(order):<30}"
          + "".join(f"{added[p]:>9}" for p in team))
print(f"{'Shapley share (average)':<30}"
      + "".join(f"{share[p]:>9.1f}" for p in team))
print(f"Shares add up to {sum(share.values()):.1f}, the whole bonus")
