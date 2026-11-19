"""Pojedynczy perceptron (Rosenblatt), zaimplementowany od zera w NumPy."""
from __future__ import annotations

import numpy as np

from .activations import step


class Perceptron:
    def __init__(self, n_inputs: int, learning_rate: float = 0.1, weights=None, bias: float = 0.0):
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
        else:
            self.weights = np.random.uniform(-1, 1, n_inputs)
        self.bias = float(bias)
        self.learning_rate = float(learning_rate)

    def net_input(self, x) -> float:
        return float(np.dot(self.weights, x) + self.bias)

    def predict(self, x) -> float:
        return step(self.net_input(x))

    def train_epoch(self, X, y) -> dict:
        """Jedna epoka klasycznej reguły uczenia perceptronu: w += lr * (t - y) * x."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        updates = 0
        for xi, target in zip(X, y):
            prediction = self.predict(xi)
            error = target - prediction
            if error != 0:
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
                updates += 1
        return {"updates": int(updates), "accuracy": self.accuracy(X, y)}

    def accuracy(self, X, y) -> float:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        preds = np.array([self.predict(xi) for xi in X])
        return float(np.mean(preds == y))

    def to_dict(self) -> dict:
        return {"weights": self.weights.tolist(), "bias": self.bias}

    @classmethod
    def from_dict(cls, data: dict, learning_rate: float = 0.1) -> "Perceptron":
        return cls(
            n_inputs=len(data["weights"]),
            learning_rate=learning_rate,
            weights=data["weights"],
            bias=data.get("bias", 0.0),
        )
