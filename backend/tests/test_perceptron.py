import pytest

from nn.datasets import get_dataset
from nn.perceptron import Perceptron


def train_to_convergence(perceptron, X, y, max_epochs=200):
    result = {"updates": None, "accuracy": None}
    for _ in range(max_epochs):
        result = perceptron.train_epoch(X, y)
        if result["updates"] == 0:
            break
    return result


def test_perceptron_learns_and():
    data = get_dataset("and")
    p = Perceptron(n_inputs=2, learning_rate=0.1, weights=[0.0, 0.0], bias=0.0)
    result = train_to_convergence(p, data["points"], data["labels"])
    assert result["accuracy"] == 1.0


def test_perceptron_learns_or():
    data = get_dataset("or")
    p = Perceptron(n_inputs=2, learning_rate=0.1, weights=[0.0, 0.0], bias=0.0)
    result = train_to_convergence(p, data["points"], data["labels"])
    assert result["accuracy"] == 1.0


def test_perceptron_cannot_learn_xor():
    data = get_dataset("xor")
    p = Perceptron(n_inputs=2, learning_rate=0.1, weights=[0.0, 0.0], bias=0.0)
    result = train_to_convergence(p, data["points"], data["labels"], max_epochs=500)
    assert result["accuracy"] < 1.0


def test_predict_matches_trained_weights():
    p = Perceptron(n_inputs=2, learning_rate=0.1, weights=[1.0, 1.0], bias=-1.5)
    assert p.predict([1, 1]) == 1.0
    assert p.predict([0, 0]) == 0.0


def test_train_epoch_trace_matches_update_rule():
    p = Perceptron(n_inputs=2, learning_rate=0.1, weights=[0.0, 0.0], bias=-0.5)
    X = [[1.0, 1.0]]
    y = [1.0]
    result = p.train_epoch(X, y, record_trace=True)

    assert result["updates"] == 1
    assert len(result["trace"]) == 1
    event = result["trace"][0]
    assert event["point"] == [1.0, 1.0]
    assert event["target"] == 1.0
    assert event["prediction"] == 0.0
    assert event["error"] == 1.0
    assert event["delta_weights"] == [0.1, 0.1]
    assert event["delta_bias"] == pytest.approx(0.1)
    assert event["weights_after"] == [0.1, 0.1]
    assert event["bias_after"] == pytest.approx(-0.4)
