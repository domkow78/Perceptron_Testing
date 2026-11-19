"""Funkcje aktywacji używane przez sieci neuronowe w tym projekcie."""
import numpy as np


def step(x):
    """Klasyczna funkcja progowa Rosenblatta: 1 jeśli x >= 0, inaczej 0."""
    return 1.0 if x >= 0 else 0.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def sigmoid_derivative(a):
    """Pochodna liczona na podstawie już obliczonej aktywacji a = sigmoid(x)."""
    return a * (1.0 - a)


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(a):
    return (a > 0.0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(a):
    return 1.0 - a**2


# Rejestr aktywacji dostępnych dla warstw ukrytych MLP: nazwa -> (funkcja, pochodna z aktywacji)
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}
