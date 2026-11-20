"""Automatyczny wybór silnika sieci (NumPy vs PyTorch) na podstawie rozmiaru
architektury — pozwala tej samej stronie frontendowej trenować zarówno małe
sieci (Etap 2, NumPy "od zera") jak i duże sieci 1000+ neuronów (Etap 3,
PyTorch) bez żadnej zmiany w JS: backend jest szczegółem implementacyjnym
serwera, nie czymś co wybiera użytkownik.
"""
from __future__ import annotations

from .mlp import MLP

TORCH_BACKEND_THRESHOLD = 256


def select_backend_name(layer_sizes) -> str:
    """"torch" dla dużych sieci (najszersza warstwa >= progu), inaczej "numpy"."""
    if max(layer_sizes) >= TORCH_BACKEND_THRESHOLD:
        return "torch"
    return "numpy"


def build_mlp(
    layer_sizes,
    hidden_activation: str = "sigmoid",
    learning_rate: float = 0.1,
    weights=None,
    biases=None,
    seed=None,
    backend: str | None = None,
):
    """Buduje sieć (świeżą albo z zapisanych wag) wybranym lub auto-dobranym
    backendem. Zwraca (siec, nazwa_backendu)."""
    backend_name = backend or select_backend_name(layer_sizes)

    if backend_name == "torch":
        from .torch_mlp import TorchMLP as _Impl
    else:
        _Impl = MLP

    if weights is not None and biases is not None:
        net = _Impl.from_dict(
            {
                "layer_sizes": layer_sizes,
                "hidden_activation": hidden_activation,
                "weights": weights,
                "biases": biases,
            },
            learning_rate=learning_rate,
        )
    else:
        net = _Impl(
            layer_sizes=layer_sizes,
            hidden_activation=hidden_activation,
            learning_rate=learning_rate,
            seed=seed,
        )
    return net, backend_name
