"""Wielowarstwowy perceptron (MLP) od zera w NumPy — propagacja w przód i wstecz.

Obsługuje dowolną liczbę warstw ukrytych o dowolnych rozmiarach. Warstwa
wyjściowa ma zawsze jeden neuron z sigmoidem (klasyfikacja binarna 0/1),
a strata to binarna entropia krzyżowa — stąd prosty wzór na błąd wyjścia
(delta = a - y), który dobrze widać w kodzie poniżej.
"""
from __future__ import annotations

import numpy as np

from .activations import ACTIVATIONS, sigmoid


class MLP:
    def __init__(
        self,
        layer_sizes,
        hidden_activation: str = "sigmoid",
        learning_rate: float = 0.1,
        weights=None,
        biases=None,
        seed=None,
    ):
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Nieznana funkcja aktywacji: {hidden_activation}")

        self.layer_sizes = list(layer_sizes)
        self.hidden_activation_name = hidden_activation
        self.hidden_fn, self.hidden_derivative = ACTIVATIONS[hidden_activation]
        self.learning_rate = float(learning_rate)

        if weights is not None and biases is not None:
            self.weights = [np.array(w, dtype=float) for w in weights]
            self.biases = [np.array(b, dtype=float) for b in biases]
        else:
            rng = np.random.default_rng(seed)
            self.weights = []
            self.biases = []
            for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
                scale = np.sqrt(1.0 / n_in)
                self.weights.append(rng.normal(0.0, scale, size=(n_in, n_out)))
                self.biases.append(np.zeros(n_out))

    def forward(self, X):
        """Zwraca listę aktywacji dla każdej warstwy, łącznie z wejściem (a[0])."""
        activations = [np.asarray(X, dtype=float)]
        a = activations[0]
        last_index = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            fn = sigmoid if i == last_index else self.hidden_fn
            a = fn(z)
            activations.append(a)
        return activations

    def predict(self, X):
        return self.forward(X)[-1]

    def train_epoch(self, X, y) -> dict:
        """Jedna epoka pełnopartyjnego spadku gradientu (backpropagation)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n_samples = X.shape[0]

        activations = self.forward(X)
        output = activations[-1]

        deltas = [output - y]  # pochodna BCE względem z przy sigmoidzie na wyjściu
        for layer in range(len(self.weights) - 2, -1, -1):
            W_next = self.weights[layer + 1]
            a = activations[layer + 1]
            delta = (deltas[0] @ W_next.T) * self.hidden_derivative(a)
            deltas.insert(0, delta)

        for layer in range(len(self.weights)):
            a_prev = activations[layer]
            grad_w = a_prev.T @ deltas[layer] / n_samples
            grad_b = deltas[layer].mean(axis=0)
            self.weights[layer] -= self.learning_rate * grad_w
            self.biases[layer] -= self.learning_rate * grad_b

        return {
            "loss": self._binary_cross_entropy(output, y),
            "accuracy": self._accuracy(output, y),
        }

    def evaluate(self, X, y) -> dict:
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        output = self.predict(X)
        return {
            "loss": self._binary_cross_entropy(output, y),
            "accuracy": self._accuracy(output, y),
        }

    @staticmethod
    def _binary_cross_entropy(output, y) -> float:
        eps = 1e-9
        clipped = np.clip(output, eps, 1 - eps)
        return float(np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))))

    @staticmethod
    def _accuracy(output, y) -> float:
        predictions = (output >= 0.5).astype(float)
        return float(np.mean(predictions == y))

    def to_dict(self) -> dict:
        return {
            "layer_sizes": self.layer_sizes,
            "hidden_activation": self.hidden_activation_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    @classmethod
    def from_dict(cls, data: dict, learning_rate: float = 0.1) -> "MLP":
        return cls(
            layer_sizes=data["layer_sizes"],
            hidden_activation=data.get("hidden_activation", "sigmoid"),
            learning_rate=learning_rate,
            weights=data["weights"],
            biases=data["biases"],
        )
