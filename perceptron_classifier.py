"""
perceptron_classifier.py
Supervised Learning: Perceptron from scratch using NumPy
Dataset: Iris flower (via sklearn.datasets)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load iris dataset
iris = load_iris()
X = iris.data    # shape (150, 4): sepal length/width, petal length/width
y = iris.target  # 0=setosa, 1=versicolor, 2=virginica

print("Iris dataset loaded.")
print(f"Features : {iris.feature_names}")
print(f"Classes  : {iris.target_names}")
print(f"Shape    : X={X.shape}, y={y.shape}")

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# perceptron
class Perceptron:

    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.lr       = learning_rate
        self.n_epochs = n_epochs
        self.weights  = None   # shape (n_classes, n_features)
        self.bias     = None   # shape (n_classes,)

    @staticmethod
    def activation(z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_  = np.unique(y)
        n_classes      = len(self.classes_)

        # Initialize weights and bias to zero
        self.weights = np.zeros((n_classes, n_features))
        self.bias    = np.zeros(n_classes)

        print(f"\n── Perceptron Hyperparameters ──")
        print(f"  Initial weights : zeros {self.weights.shape}")
        print(f"  Initial bias    : zeros {self.bias.shape}")
        print(f"  Learning rate   : {self.lr}")
        print(f"  Epochs          : {self.n_epochs}")
        print(f"  Activation      : Step function (threshold=0)")
        print(f"  Strategy        : One-vs-Rest (one perceptron per class)")

        # One-vs-Rest: train one binary perceptron per class
        for epoch in range(self.n_epochs):
            for c_idx, c in enumerate(self.classes_):
                # Binary label: 1 if sample belongs to class c, else 0
                y_binary = (y == c).astype(int)

                for i in range(n_samples):
                    z         = np.dot(self.weights[c_idx], X[i]) + self.bias[c_idx]
                    y_hat     = self.activation(z)
                    error     = y_binary[i] - y_hat

                    # Perceptron update rule: w = w + lr * error * x
                    self.weights[c_idx] += self.lr * error * X[i]
                    self.bias[c_idx]    += self.lr * error

    def predict(self, X):
        """Return class with highest raw score (w·x + b) for each sample."""
        scores = X @ self.weights.T + self.bias   # shape (n_samples, n_classes)
        return self.classes_[np.argmax(scores, axis=1)]

# training and evaluation
perceptron = Perceptron(learning_rate=0.1, n_epochs=100)
perceptron.fit(X_train, y_train)

y_train_pred = perceptron.predict(X_train)
y_test_pred  = perceptron.predict(X_test)

train_acc = np.mean(y_train_pred == y_train) * 100
test_acc  = np.mean(y_test_pred  == y_test)  * 100

print(f"\nTraining Accuracy : {train_acc:.2f}%")
print(f"Testing  Accuracy : {test_acc:.2f}%")

print("\nFinal weights per class:")
for i, cls in enumerate(perceptron.classes_):
    print(f"  Class {cls} ({iris.target_names[cls]}): w={perceptron.weights[i]}, b={perceptron.bias[i]:.4f}")