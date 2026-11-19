"""Proste, 2D datasety do zabawy z perceptronem i (docelowo) sieciami neuronowymi."""
from __future__ import annotations

import numpy as np

_LOGIC_GATES = {
    "and": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "or": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "xor": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
}


def _noisy_logic_gate(name: str, points_per_corner: int = 20, noise: float = 0.08, seed: int = 42):
    rng = np.random.default_rng(seed)
    corners = _LOGIC_GATES[name]
    xs, ys = [], []
    for (cx, cy), label in corners.items():
        pts = rng.normal(loc=[cx, cy], scale=noise, size=(points_per_corner, 2))
        xs.append(pts)
        ys.extend([label] * points_per_corner)
    return np.vstack(xs), np.array(ys, dtype=float)


def _blobs(n_per_class: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=[0.25, 0.75], scale=0.12, size=(n_per_class, 2))
    c1 = rng.normal(loc=[0.75, 0.25], scale=0.12, size=(n_per_class, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=float)
    return X, y


def get_dataset(name: str) -> dict:
    """Zwraca dataset jako {"points": [[x, y], ...], "labels": [0/1, ...]}."""
    key = name.lower()
    if key in _LOGIC_GATES:
        X, y = _noisy_logic_gate(key)
    elif key == "blobs":
        X, y = _blobs()
    else:
        raise ValueError(f"Nieznany dataset: {name}")
    return {"points": X.tolist(), "labels": y.tolist()}
