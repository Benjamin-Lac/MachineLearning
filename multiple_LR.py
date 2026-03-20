import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# loading gas properties
df = pd.read_csv("GasProperties.csv")
print("Dataset preview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

target_col = df.columns[-1]
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(float)

#RMSE helper function
def rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

# least squares solution: w_hat = (X^T X)^{-1} X^T y
def least_squares(X_b, y):
    X_transpose = np.transpose(X_b)
    xtx = np.dot(X_transpose, X_b)
    xty = np.dot(X_transpose, y)
    xtx_inv = np.linalg.inv(xtx)
    w_hat = np.dot(xtx_inv, xty)
    return w_hat

def add_bias_column(X):
    N = X.shape[0]
    return np.hstack([np.ones((N, 1)), X])

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# original data training
X_train_b = add_bias_column(X_train)
X_test_b  = add_bias_column(X_test)

w_hat = least_squares(X_train_b, y_train)

y_train_pred = np.dot(X_train_b, w_hat)
y_test_pred  = np.dot(X_test_b, w_hat)

train_rmse_orig = rmse(y_train, y_train_pred)
test_rmse_orig  = rmse(y_test,  y_test_pred)

print("\n── Without Normalization ──")
print(f"Learned weights (w0=intercept, w1..wN=features): {w_hat}")
print(f"Training RMSE : {train_rmse_orig:.4f}")
print(f"Testing  RMSE : {test_rmse_orig:.4f}")

# z score normalization and outlier removal
norm_cols = ["T", "P", "TC", "SV"]
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

# train on normalized data
X_norm = df_norm[feature_cols].values.astype(float)
y_norm = df_norm[target_col].values.astype(float)

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_norm, y_norm, test_size=0.2, random_state=42
)

X_train_nb = add_bias_column(X_train_n)
X_test_nb  = add_bias_column(X_test_n)

w_hat_n = least_squares(X_train_nb, y_train_n)

y_train_pred_n = np.dot(X_train_nb, w_hat_n)
y_test_pred_n  = np.dot(X_test_nb, w_hat_n)

train_rmse_norm = rmse(y_train_n, y_train_pred_n)
test_rmse_norm  = rmse(y_test_n,  y_test_pred_n)

print("\n── With Z-score Normalization ──")
print(f"Learned weights: {w_hat_n}")
print(f"Training RMSE : {train_rmse_norm:.4f}")
print(f"Testing  RMSE : {test_rmse_norm:.4f}")

# comparison
print("\n── RMSE Comparison ──")
print(f"{'':30s} {'Train RMSE':>12} {'Test RMSE':>12}")
print(f"{'Without normalization':30s} {train_rmse_orig:>12.4f} {test_rmse_orig:>12.4f}")
print(f"{'With normalization':30s} {train_rmse_norm:>12.4f} {test_rmse_norm:>12.4f}")

improved = test_rmse_norm < test_rmse_orig
print(f"\nNormalization improved accuracy? {'YES' if improved else 'NO'}")
