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

    def train_epoch(self, X, y, record_trace: bool = False) -> dict:
        """Jedna epoka klasycznej reguły uczenia perceptronu: w += lr * (t - y) * x.

        Z `record_trace=True` zwraca też listę zdarzeń (jedno na błędnie
        sklasyfikowany punkt) z pełnymi liczbami użytymi do aktualizacji wag —
        to surowy materiał na "żywy log" pokazywany w UI.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        updates = 0
        trace = [] if record_trace else None
        for xi, target in zip(X, y):
            net_input = self.net_input(xi)
            prediction = step(net_input)
            error = target - prediction
            if error != 0:
                delta_weights = self.learning_rate * error * xi
                delta_bias = self.learning_rate * error
                self.weights += delta_weights
                self.bias += delta_bias
                updates += 1
                if record_trace:
                    trace.append(
                        {
                            "point": xi.tolist(),
                            "target": float(target),
                            "prediction": float(prediction),
                            "net_input": float(net_input),
                            "error": float(error),
                            "delta_weights": delta_weights.tolist(),
                            "delta_bias": float(delta_bias),
                            "weights_after": self.weights.tolist(),
                            "bias_after": float(self.bias),
                        }
                    )
        result = {"updates": int(updates), "accuracy": self.accuracy(X, y)}
        if record_trace:
            result["trace"] = trace
        return result

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
