"""Funkcje aktywacji używane przez sieci neuronowe w tym projekcie."""


def step(x):
    """Klasyczna funkcja progowa Rosenblatta: 1 jeśli x >= 0, inaczej 0."""
    return 1.0 if x >= 0 else 0.0
