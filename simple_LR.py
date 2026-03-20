import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("study_data.csv")
df.columns = df.columns.str.strip().str.lower()

X = df[["hours"]].values
y = df["score"].values

def rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

print("(a) Splitting dataset into 80% training and 20% testing set")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Training samples: {len(X_train)} (80%)")
print(f"    Testing samples: {len(X_test)} (20%)")

print("\n(b) Fitting linear regression model using LinearRegression()")
model = LinearRegression()
model.fit(X_train, y_train)

w0 = model.intercept_
w1 = model.coef_[0]

print("\n(c) Computing training and testing RMSE")
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_rmse = rmse(y_train, y_train_pred)
test_rmse  = rmse(y_test,  y_test_pred)

print(f"    Training RMSE: {train_rmse:.4f}")
print(f"    Testing RMSE: {test_rmse:.4f}")

print("\n(d) Learned function in polynomial form:")
print(f"    y = {w0:.4f} + {w1:.4f} * x")
