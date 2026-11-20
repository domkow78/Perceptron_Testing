"""Wielowarstwowy perceptron o tym samym interfejsie co MLP, ale liczony
przez PyTorch — ``nn.Linear`` i autograd zamiast ręcznie wyprowadzonego
backpropu. Ten sam algorytm (pełnopartyjny spadek gradientu, binarna
entropia krzyżowa, sigmoid na wyjściu), tylko gradienty liczy biblioteka
zamiast ręcznych wzorów — to pozwala skalować się do dużo większych sieci
(1000+ neuronów), gdzie ręczny NumPy zaczyna być niewygodny.

Uwaga: inicjalizacja wag powtarza ten sam schemat co MLP (rozkład normalny
ze skalą sqrt(1/n_in), zerowe biasy), ale z własnym generatorem PyTorcha —
przy tym samym ``seed`` wartości wag NIE będą identyczne jak w MLP (inny
silnik losowy), co jest świadome i nieistotne dla wyniku uczenia.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_TORCH_HIDDEN = {
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "tanh": torch.tanh,
}


class TorchMLP:
    def __init__(
        self,
        layer_sizes,
        hidden_activation: str = "sigmoid",
        learning_rate: float = 0.1,
        weights=None,
        biases=None,
        seed=None,
    ):
        if hidden_activation not in _TORCH_HIDDEN:
            raise ValueError(f"Nieznana funkcja aktywacji: {hidden_activation}")

        self.layer_sizes = list(layer_sizes)
        self.hidden_activation_name = hidden_activation
        self.hidden_fn = _TORCH_HIDDEN[hidden_activation]
        self.learning_rate = float(learning_rate)

        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out, dtype=torch.float64)
            for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        )

        if weights is not None and biases is not None:
            with torch.no_grad():
                for layer, w, b in zip(self.layers, weights, biases):
                    # kontrakt JSON ma macierz (n_in, n_out), nn.Linear.weight ma (n_out, n_in)
                    layer.weight.copy_(torch.as_tensor(np.array(w, dtype=np.float64)).T)
                    layer.bias.copy_(torch.as_tensor(np.array(b, dtype=np.float64)))
        else:
            gen = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
            with torch.no_grad():
                for layer, n_in in zip(self.layers, self.layer_sizes[:-1]):
                    scale = (1.0 / n_in) ** 0.5
                    layer.weight.normal_(mean=0.0, std=scale, generator=gen)
                    layer.bias.zero_()

    def forward(self, X):
        """Zwraca listę aktywacji (jako tablice numpy) dla każdej warstwy, łącznie z wejściem."""
        x = torch.as_tensor(np.asarray(X, dtype=np.float64))
        activations = [x]
        last_index = len(self.layers) - 1
        with torch.no_grad():
            a = x
            for i, layer in enumerate(self.layers):
                z = layer(a)
                fn = torch.sigmoid if i == last_index else self.hidden_fn
                a = fn(z)
                activations.append(a)
        return [a.numpy() for a in activations]

    def predict(self, X):
        return self.forward(X)[-1]

    def train_epoch(self, X, y) -> dict:
        """Jedna epoka pełnopartyjnego spadku gradientu, gradienty z autogradu."""
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float64))
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64)).reshape(-1, 1)

        a = X_t
        last_index = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            z = layer(a)
            fn = torch.sigmoid if i == last_index else self.hidden_fn
            a = fn(z)
        output = a

        eps = 1e-9
        clipped = output.clamp(eps, 1 - eps)
        loss = -(y_t * clipped.log() + (1 - y_t) * (1 - clipped).log()).mean()

        for layer in self.layers:
            layer.weight.grad = None
            layer.bias.grad = None
        loss.backward()

        with torch.no_grad():
            for layer in self.layers:
                layer.weight -= self.learning_rate * layer.weight.grad
                layer.bias -= self.learning_rate * layer.bias.grad

        return {
            "loss": float(loss.item()),
            "accuracy": self._accuracy(output.detach().numpy(), y_t.numpy()),
        }

    def evaluate(self, X, y) -> dict:
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        output = self.predict(X)
        return {
            "loss": self._binary_cross_entropy(output, y_arr),
            "accuracy": self._accuracy(output, y_arr),
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
        with torch.no_grad():
            weights = [layer.weight.detach().numpy().T.tolist() for layer in self.layers]
            biases = [layer.bias.detach().numpy().tolist() for layer in self.layers]
        return {
            "layer_sizes": self.layer_sizes,
            "hidden_activation": self.hidden_activation_name,
            "weights": weights,
            "biases": biases,
        }

    @classmethod
    def from_dict(cls, data: dict, learning_rate: float = 0.1) -> "TorchMLP":
        return cls(
            layer_sizes=data["layer_sizes"],
            hidden_activation=data.get("hidden_activation", "sigmoid"),
            learning_rate=learning_rate,
            weights=data["weights"],
            biases=data["biases"],
        )
