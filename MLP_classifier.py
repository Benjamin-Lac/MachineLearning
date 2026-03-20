"""
MLP_classifier.py
Supervised Learning: Multi-Layer Perceptron (MLP) using scikit-learn
Dataset: Iris flower (via sklearn.datasets)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def accuracy(model, X, y):
    return model.score(X, y) * 100


#baseline
print("Part (a): Baseline MLP — 1 hidden layer, 3 neurons")

mlp_base = MLPClassifier(
    hidden_layer_sizes=(3,),   # 1 hidden layer with 3 neurons
    activation="relu",         # activation function
    solver="adam",             # optimizer (handles learning rate adaptively)
    learning_rate_init=0.001,  # initial learning rate
    max_iter=1000,             # number of epochs
    batch_size=32,             # mini-batch size
    random_state=42
)
mlp_base.fit(X_train, y_train)

print("\nHyperparameters:")
print(f"  hidden_layer_sizes : (3,)   — 1 hidden layer with 3 neurons")
print(f"  Output neurons     : 3      — one per class (softmax internally)")
print(f"  Activation         : ReLU")
print(f"  Optimizer/solver   : Adam")
print(f"  Initial LR         : 0.001")
print(f"  Max epochs         : 1000")
print(f"  Batch size         : 32")
print(f"  Initial weights    : Glorot uniform (sklearn default)")
print(f"  Initial bias       : zeros (sklearn default)")

train_acc_base = accuracy(mlp_base, X_train, y_train)
test_acc_base  = accuracy(mlp_base, X_test,  y_test)

print(f"\nTraining Accuracy : {train_acc_base:.2f}%")
print(f"Testing  Accuracy : {test_acc_base:.2f}%")

# deeper/ wider 
print("Part (b): Improved MLP — deeper architecture")

mlp_deep = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=2000,
    batch_size=32,
    random_state=42
)
mlp_deep.fit(X_train, y_train)

print("\nImproved architecture:")
print(f"  hidden_layer_sizes : (64, 32, 16)  — 3 hidden layers")
print(f"  Activation         : ReLU")
print(f"  Max epochs         : 2000")

train_acc_deep = accuracy(mlp_deep, X_train, y_train)
test_acc_deep  = accuracy(mlp_deep, X_test,  y_test)

print(f"\nTraining Accuracy : {train_acc_deep:.2f}%")
print(f"Testing  Accuracy : {test_acc_deep:.2f}%")

# results
print("\n── Accuracy Comparison ──")
print(f"{'Model':35s} {'Train Acc':>10} {'Test Acc':>10}")
print(f"{'Baseline MLP (1 hidden, 3 neurons)':35s} {train_acc_base:>9.2f}% {test_acc_base:>9.2f}%")
print(f"{'Improved MLP (64-32-16 neurons)':35s} {train_acc_deep:>9.2f}% {test_acc_deep:>9.2f}%")

improved = test_acc_deep > test_acc_base
print(f"\nImproved classification performance? {'YES' if improved else 'NO (already near perfect)'}")