# Exercise 3: the fraud estimate for an issuer ten times the size.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "estimates", "code/26/01_estimates.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)         # defines Workload; prints nothing

cards, fraud = 80e6, 0.001
variants = {
    "keep 1% legit": est.Workload(
        "a", cards, 1, 4, 0.5, 15, 300, 150e6, 1,
        90 * cards * (fraud + 0.01), 4),
    "keep 10% legit": est.Workload(
        "b", cards, 1, 4, 0.5, 15, 300, 150e6, 1,
        90 * cards * (fraud + 0.10), 4),
}
table = {k: v.estimate() for k, v in variants.items()}
print(f"{'':20}" + "".join(f"{k:>16}" for k in table))
for metric in next(iter(table.values())):
    print(f"{metric:20}" + "".join(
        f"{est.show(t[metric]):>16}" for t in table.values()))
