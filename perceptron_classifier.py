import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

print("(a) Splitting dataset into 80/20")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"    Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

print("(b) Implementing Perceptron from scratch")
class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    @staticmethod
    def activation(z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for epoch in range(self.n_epochs):
            for c_idx, c in enumerate(self.classes_):
                y_binary = (y == c).astype(int)

                for i in range(n_samples):
                    z = np.dot(self.weights[c_idx], X[i]) + self.bias[c_idx]
                    y_hat = self.activation(z)
                    error = y_binary[i] - y_hat

                    self.weights[c_idx] += self.lr * error * X[i]
                    self.bias[c_idx] += self.lr * error

    def predict(self, X):
        scores = X @ self.weights.T + self.bias
        return self.classes_[np.argmax(scores, axis=1)]

perceptron = Perceptron(learning_rate=0.1, n_epochs=100)
perceptron.fit(X_train, y_train)

y_train_pred = perceptron.predict(X_train)
y_test_pred = perceptron.predict(X_test)

train_acc = np.mean(y_train_pred == y_train) * 100
test_acc = np.mean(y_test_pred == y_test) * 100

print(f"    Training Accuracy: {train_acc:.2f}%")
print(f"    Testing Accuracy: {test_acc:.2f}%")

print("\n    Final weights per class:")
for i, cls in enumerate(perceptron.classes_):
    print(f"      Class {cls}: w={perceptron.weights[i]}, b={perceptron.bias[i]:.4f}")