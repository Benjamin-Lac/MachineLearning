"""
multiple_LR.py
Supervised Learning: Multiple Linear Regression (from scratch with NumPy)
Dataset: GasProperties.csv
Goal: Predict gas quality from chemical properties.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("GasProperties.csv")
print("Dataset preview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── 2. Prepare features and target ───────────────────────────────────────────
# Assumes last column is the target (gas quality) and the rest are features.
# Adjust column names below if needed.
target_col = df.columns[-1]
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(float)

# ── 3. RMSE helper ────────────────────────────────────────────────────────────
def rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

# ── 4. Least squares solver (from scratch) ────────────────────────────────────
def least_squares(X_b, y):
    """
    Compute w_hat = (X^T X)^{-1} X^T y
    X_b should already have the bias column of ones prepended.
    """
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

def add_bias_column(X):
    """Prepend a column of ones for the intercept term."""
    N = X.shape[0]
    return np.hstack([np.ones((N, 1)), X])

# ── 5. Split 80/20 ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 6a. Train on ORIGINAL data ────────────────────────────────────────────────
X_train_b = add_bias_column(X_train)
X_test_b  = add_bias_column(X_test)

w_hat = least_squares(X_train_b, y_train)

y_train_pred = X_train_b @ w_hat
y_test_pred  = X_test_b  @ w_hat

train_rmse_orig = rmse(y_train, y_train_pred)
test_rmse_orig  = rmse(y_test,  y_test_pred)

print("\n── Without Normalization ──")
print(f"Learned weights (w0=intercept, w1..wN=features): {w_hat}")
print(f"Training RMSE : {train_rmse_orig:.4f}")
print(f"Testing  RMSE : {test_rmse_orig:.4f}")

# ── 6b. Z-score normalization + outlier removal ───────────────────────────────
norm_cols = ["T", "P", "TC", "SV"]   # columns to normalize (adjust if needed)
norm_cols = [c for c in norm_cols if c in df.columns]

df_norm = df.copy()

for col in norm_cols:
    mean = df_norm[col].mean()
    std  = df_norm[col].std()
    df_norm[col] = (df_norm[col] - mean) / std   # z-score

# Remove rows where any normalized column is outside (-2, +2)
mask = (df_norm[norm_cols].abs() <= 2).all(axis=1)
df_norm = df_norm[mask].reset_index(drop=True)

print(f"\nRows before outlier removal: {len(df)}")
print(f"Rows after  outlier removal: {len(df_norm)}")

# Save normalized dataset
df_norm.to_csv("GasProperties_norm.csv", index=False)
print("Normalized dataset saved to GasProperties_norm.csv")

# ── 7. Train on NORMALIZED data ───────────────────────────────────────────────
X_norm = df_norm[feature_cols].values.astype(float)
y_norm = df_norm[target_col].values.astype(float)

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_norm, y_norm, test_size=0.2, random_state=42
)

X_train_nb = add_bias_column(X_train_n)
X_test_nb  = add_bias_column(X_test_n)

w_hat_n = least_squares(X_train_nb, y_train_n)

y_train_pred_n = X_train_nb @ w_hat_n
y_test_pred_n  = X_test_nb  @ w_hat_n

train_rmse_norm = rmse(y_train_n, y_train_pred_n)
test_rmse_norm  = rmse(y_test_n,  y_test_pred_n)

print("\n── With Z-score Normalization ──")
print(f"Learned weights: {w_hat_n}")
print(f"Training RMSE : {train_rmse_norm:.4f}")
print(f"Testing  RMSE : {test_rmse_norm:.4f}")

# ── 8. Comparison summary ─────────────────────────────────────────────────────
print("\n── RMSE Comparison ──")
print(f"{'':30s} {'Train RMSE':>12} {'Test RMSE':>12}")
print(f"{'Without normalization':30s} {train_rmse_orig:>12.4f} {test_rmse_orig:>12.4f}")
print(f"{'With normalization':30s} {train_rmse_norm:>12.4f} {test_rmse_norm:>12.4f}")

improved = test_rmse_norm < test_rmse_orig
print(f"\nNormalization improved accuracy? {'YES' if improved else 'NO'}")
print("""
Explanation:
  Z-score normalization scales all features to a comparable range (mean=0, std=1).
  This prevents features with large raw magnitudes (e.g. pressure P) from dominating
  the regression coefficients. Removing outliers (|z| > 2) reduces noise that would
  otherwise distort the least-squares solution, leading to lower RMSE.
""")