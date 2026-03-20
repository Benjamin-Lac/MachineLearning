import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("GasProperties.csv")

target_col = df.columns[-1]
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(float)

def rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

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

print("(a) Splitting dataset into 80% training and 20% testing set")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Training samples: {len(X_train)} (80%)")
print(f"    Testing samples: {len(X_test)} (20%)")

print("\n(b) Implementing least squares method from scratch using NumPy")
X_train_b = add_bias_column(X_train)
X_test_b  = add_bias_column(X_test)

w_hat = least_squares(X_train_b, y_train)
print(f"    Computed coefficient vector w_hat: {w_hat}")

print("\n(c) Computing training and testing RMSE")
y_train_pred = np.dot(X_train_b, w_hat)
y_test_pred  = np.dot(X_test_b, w_hat)

train_rmse_orig = rmse(y_train, y_train_pred)
test_rmse_orig  = rmse(y_test,  y_test_pred)

print(f"    Training RMSE: {train_rmse_orig:.4f}")
print(f"    Testing RMSE: {test_rmse_orig:.4f}")

print("\n(d) Normalizing variables T, P, TC, SV using z-score and removing outliers")
# z score normalization and outlier removal
norm_cols = ["T", "P", "TC", "SV"]
norm_cols = [c for c in norm_cols if c in df.columns]

df_norm = df.copy()

for col in norm_cols:
    mean = df_norm[col].mean()
    std  = df_norm[col].std()
    df_norm[col] = (df_norm[col] - mean) / std

mask = (df_norm[norm_cols].abs() <= 2).all(axis=1)
print(f"    Rows before outlier removal: {len(df)}")
df_norm = df_norm[mask].reset_index(drop=True)
print(f"    Rows after outlier removal: {len(df_norm)}")

df_norm.to_csv("GasProperties_norm.csv", index=False)
print(f"    Saved normalized dataset to GasProperties_norm.csv")

X_norm = df_norm[feature_cols].values.astype(float)
y_norm = df_norm[target_col].values.astype(float)

print("\n(e) Retraining linear regression model using normalized dataset")
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_norm, y_norm, test_size=0.2, random_state=42
)

X_train_nb = add_bias_column(X_train_n)
X_test_nb  = add_bias_column(X_test_n)

w_hat_n = least_squares(X_train_nb, y_train_n)
print(f"    Computed coefficient vector w_hat: {w_hat_n}")

y_train_pred_n = np.dot(X_train_nb, w_hat_n)
y_test_pred_n  = np.dot(X_test_nb, w_hat_n)

train_rmse_norm = rmse(y_train_n, y_train_pred_n)
test_rmse_norm  = rmse(y_test_n,  y_test_pred_n)

print(f"    Training RMSE: {train_rmse_norm:.4f}")
print(f"    Testing RMSE: {test_rmse_norm:.4f}")

print("\n(f) Comparing RMSE before and after normalization")
print(f"Without normalization - Training RMSE: {train_rmse_orig:.4f}, Testing RMSE: {test_rmse_orig:.4f}")
print(f"With normalization - Training RMSE: {train_rmse_norm:.4f}, Testing RMSE: {test_rmse_norm:.4f}")

improved = test_rmse_norm < test_rmse_orig
print(f"Normalization improved accuracy? {'YES' if improved else 'NO'}")
