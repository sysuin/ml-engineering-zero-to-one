# A logistic model's score is a dot product; many scores, a matrix.
import numpy as np

from foresight.models.logistic import (TRAIN, RenewalRisk, features,
                                       load, sigmoid)

train = load(*TRAIN)
model = RenewalRisk().fit(train)
w = model.model_.coef_[0]                 # one weight per column
b = model.model_.intercept_[0]
Z = model.scaler_.transform(features(train).to_numpy())

# One contract: the one the model ranks riskiest.
p_library = model.predict_proba(train)
i = int(np.argmax(p_library))
x = Z[i]
row = train.iloc[i]
print(f"Contract {row.contract_id}, {row.segment}, {row.region}:"
      f" {len(x)} numbers in, one out")
parts = w * x                             # w_j times x_j, per column
order = np.argsort(-np.abs(parts))
print(f"{'column':<22}{'x_j':>8}{'w_j':>8}{'w_j x_j':>9}")
for j in order[:5]:
    print(f"{model.columns_[j]:<22}{x[j]:>8.2f}{w[j]:>8.3f}"
          f"{parts[j]:>9.3f}")
rest = f"the other {len(x) - 5}"
print(f"{rest:<38}{parts[order[5:]].sum():>9.3f}")
z = w @ x + b
print(f"w . x = {w @ x:.3f}, plus b = {b:.3f}, gives z = {z:.3f}")
print(f"sigmoid(z) = {sigmoid(z):.4f};"
      f" the library says {p_library[i]:.4f}")

# Every contract at once: a matrix times a vector.
scores = sigmoid(Z @ w + b)
print(f"\nZ is {Z.shape[0]:,} x {Z.shape[1]}, w has {w.shape[0]}:"
      f" Z @ w has {scores.shape[0]:,} entries")
print(f"Largest difference from the library"
      f" {np.abs(scores - p_library).max():.1e}")

# The transpose runs the other way: one number per column, summed
# over contracts. It is how the gradient of Chapter 7 is computed.
error = scores - train.not_renewed.to_numpy()
gradient = Z.T @ error / len(error)
print(f"Z.T @ error has {gradient.shape[0]} entries; largest size"
      f" {np.abs(gradient).max():.1e}")
