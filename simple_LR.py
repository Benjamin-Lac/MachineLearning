import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
#loading data
df = pd.read_csv("study_data.csv")
# normalzie for white space and case
df.columns = df.columns.str.strip().str.lower()
print("Dataset preview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
 
X = df[["hours"]].values   # 2-D array required by sklearn
y = df["score"].values     # 1-D target array
 
#80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")
 
#linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
w0 = model.intercept_     # bias / intercept
w1 = model.coef_[0]       # slope
 
# rmse helper function
def rmse(y_actual, y_predicted):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))
 
# compute predictions and RMSE for train and test sets
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)
 
train_rmse = rmse(y_train, y_train_pred)
test_rmse  = rmse(y_test,  y_test_pred)
 
print(f"\nTraining RMSE : {train_rmse:.4f}")
print(f"Testing  RMSE : {test_rmse:.4f}")
 
# print learned function
print(f"\nLearned function:")
print(f"  y = {w0:.4f} + {w1:.4f} * x")
 
# plot data points and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="steelblue", label="Train data", alpha=0.7)
plt.scatter(X_test,  y_test,  color="orange",    label="Test data",  alpha=0.7)
 
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.plot(x_line, model.predict(x_line), color="red", linewidth=2,
         label=f"y = {w0:.2f} + {w1:.2f}x")
 
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("simple_LR_plot.png", dpi=150)
plt.show()
print("\nPlot saved to simple_LR_plot.png")