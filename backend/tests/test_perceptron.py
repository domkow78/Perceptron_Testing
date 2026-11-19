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
