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


def _circles(n_per_class: int = 60, seed: int = 42):
    """Dwa współśrodkowe okręgi — nie da się ich rozdzielić jedną prostą."""
    rng = np.random.default_rng(seed)

    angles_in = rng.uniform(0, 2 * np.pi, n_per_class)
    radius_in = rng.normal(0.15, 0.025, n_per_class)
    inner = np.stack([0.5 + radius_in * np.cos(angles_in), 0.5 + radius_in * np.sin(angles_in)], axis=1)

    angles_out = rng.uniform(0, 2 * np.pi, n_per_class)
    radius_out = rng.normal(0.42, 0.025, n_per_class)
    outer = np.stack([0.5 + radius_out * np.cos(angles_out), 0.5 + radius_out * np.sin(angles_out)], axis=1)

    X = np.vstack([inner, outer])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=float)
    return X, y


def _spiral(n_per_class: int = 80, seed: int = 42, turns: float = 1.6):
    """Dwie przeplatające się spirale — klasyczny test dla sieci wielowarstwowych."""
    rng = np.random.default_rng(seed)

    def make_arm(label, angle_offset):
        t = np.linspace(0.08, 1.0, n_per_class)
        r = t * 0.42
        theta = t * turns * 2 * np.pi + angle_offset + rng.normal(0, 0.06, n_per_class)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        return np.stack([x, y], axis=1), np.full(n_per_class, label, dtype=float)

    X0, y0 = make_arm(0.0, 0.0)
    X1, y1 = make_arm(1.0, np.pi)
    return np.vstack([X0, X1]), np.concatenate([y0, y1])


def get_dataset(name: str) -> dict:
    """Zwraca dataset jako {"points": [[x, y], ...], "labels": [0/1, ...]}."""
    key = name.lower()
    if key in _LOGIC_GATES:
        X, y = _noisy_logic_gate(key)
    elif key == "blobs":
        X, y = _blobs()
    elif key == "circles":
        X, y = _circles()
    elif key == "spiral":
        X, y = _spiral()
    else:
        raise ValueError(f"Nieznany dataset: {name}")
    return {"points": X.tolist(), "labels": y.tolist()}
