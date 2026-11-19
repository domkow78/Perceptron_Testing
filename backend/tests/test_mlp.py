import numpy as np

from nn.datasets import get_dataset
from nn.mlp import MLP


def train(net, X, y, epochs):
    for _ in range(epochs):
        net.train_epoch(X, y)


def test_mlp_learns_xor_with_hidden_layer():
    data = get_dataset("xor")
    net = MLP(layer_sizes=[2, 8, 1], hidden_activation="tanh", learning_rate=0.5, seed=1)
    train(net, data["points"], data["labels"], epochs=2000)
    result = net.evaluate(data["points"], data["labels"])
    assert result["accuracy"] >= 0.95


def test_forward_shapes_for_multiple_hidden_layers():
    net = MLP(layer_sizes=[2, 5, 3, 1], seed=0)
    activations = net.forward([[0.1, 0.2], [0.3, 0.4]])
    assert [a.shape for a in activations] == [(2, 2), (2, 5), (2, 3), (2, 1)]


def test_output_is_a_probability():
    net = MLP(layer_sizes=[2, 4, 1], seed=0)
    output = net.predict([[0.1, 0.9], [0.5, 0.5]])
    assert np.all(output >= 0.0) and np.all(output <= 1.0)


def test_serialization_roundtrip():
    net = MLP(layer_sizes=[2, 4, 1], hidden_activation="relu", seed=0)
    data = net.to_dict()
    restored = MLP.from_dict(data)
    X = [[0.2, 0.7], [0.9, 0.1]]
    np.testing.assert_allclose(net.predict(X), restored.predict(X))


def test_train_epoch_reduces_loss_on_separable_data():
    data = get_dataset("blobs")
    net = MLP(layer_sizes=[2, 10, 1], learning_rate=0.5, seed=2)
    before = net.evaluate(data["points"], data["labels"])["loss"]
    train(net, data["points"], data["labels"], epochs=100)
    after = net.evaluate(data["points"], data["labels"])["loss"]
    assert after < before
