import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X = iris.data
y = iris.target

print("Loading dataset and performing 80/20 split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"    Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

def accuracy(model, X, y):
    return model.score(X, y) * 100

print("\n(a) Training baseline MLP (1 hidden layer, 3 neurons)")
mlp_base = MLPClassifier(
    hidden_layer_sizes=(3,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    batch_size=32,
    random_state=42
)
mlp_base.fit(X_train, y_train)

train_acc_base = accuracy(mlp_base, X_train, y_train)
test_acc_base = accuracy(mlp_base, X_test, y_test)

print(f"    Training Accuracy: {train_acc_base:.2f}%")
print(f"    Testing Accuracy: {test_acc_base:.2f}%")

print("\n(b) Training improved MLP")
mlp_deep = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=2000,
    batch_size=32,
    random_state=42
)
mlp_deep.fit(X_train, y_train)

train_acc_deep = accuracy(mlp_deep, X_train, y_train)
test_acc_deep = accuracy(mlp_deep, X_test, y_test)

print(f"    Training Accuracy: {train_acc_deep:.2f}%")
print(f"    Testing Accuracy: {test_acc_deep:.2f}%")

print("\nComparing models:")
print(f"Baseline MLP (1 hidden, 3 neurons) - Train: {train_acc_base:.2f}%, Test: {test_acc_base:.2f}%")
print(f"Improved MLP (64-32-16 neurons) - Train: {train_acc_deep:.2f}%, Test: {test_acc_deep:.2f}%")

improved = test_acc_deep > test_acc_base
print(f"Improved performance? {'YES' if improved else 'NO (already near perfect)'}")