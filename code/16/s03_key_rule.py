# Exercise 3: a rule, not a model, for key accounts. Flag a key-account
# contract whose complaints in the 90 days before its mark are at least
# three, and at least three times its own rate over the year before.
from foresight.features.build import load
from foresight.slices import key_accounts

table = load()
rows = table[table.end_date.between("2023-01-01", "2024-12-31")
             & table.account_id.isin(key_accounts())]
before = (rows.complaints_365 - rows.complaints_90d) * 90 / 275
flag = (rows.complaints_90d >= 3) & (rows.complaints_90d >= 3 * before)
print(f"Key-account contracts ending 2023-2024: {len(rows)},"
      f" leavers {rows.not_renewed.sum()}")
print(f"Flagged by the rule: {flag.sum()}")
print(f"{'contract':>10}{'mark':>12}{'90 days':>9}{'year, per 90':>14}"
      f"{'left':>6}")
for i in rows.index[flag]:
    r = rows.loc[i]
    print(f"{r.contract_id:>10}{str(r.moment.date()):>12}"
          f"{r.complaints_90d:>9.0f}{before[i]:>14.1f}"
          f"{r.not_renewed:>6}")
most = rows.complaints_90d.sort_values(ascending=False).head(4)
print("Most complaints in 90 days:",
      ", ".join(f"{v:.0f}" for v in most))
