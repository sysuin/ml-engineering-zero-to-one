# October's list as v0.6 makes it: each call with its chance, a mark
# below the break-even, three reasons in words, and any value the model
# has rarely seen. Shown: Colm Reyes's two calls and the last call.
from foresight.score import page, score

result = score("2024-10-02", checks=False)
listed = result["list"]
mine = listed[listed.account_manager == "Colm Reyes"]["rank"]
print(page(result, ranks=list(mine) + [len(listed)]))
