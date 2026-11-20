import pytest

from nn.backend_select import TORCH_BACKEND_THRESHOLD, build_mlp, select_backend_name
from nn.mlp import MLP
from nn.torch_mlp import TorchMLP


def test_small_architecture_selects_numpy():
    assert select_backend_name([2, 10, 1]) == "numpy"


def test_large_architecture_selects_torch():
    assert select_backend_name([2, TORCH_BACKEND_THRESHOLD, 1]) == "torch"


def test_just_below_threshold_selects_numpy():
    assert select_backend_name([2, TORCH_BACKEND_THRESHOLD - 1, 1]) == "numpy"


def test_build_mlp_returns_matching_instance_type():
    net, name = build_mlp([2, 4, 1], seed=0)
    assert name == "numpy" and isinstance(net, MLP)

    net, name = build_mlp([2, TORCH_BACKEND_THRESHOLD, 1], seed=0)
    assert name == "torch" and isinstance(net, TorchMLP)


def test_build_mlp_explicit_backend_override():
    net, name = build_mlp([2, 4, 1], seed=0, backend="torch")
    assert name == "torch" and isinstance(net, TorchMLP)


@pytest.mark.parametrize("layer_sizes", [[2, 4, 1], [2, 10, 10, 1], [3, 50, 20, 2]])
def test_to_dict_shape_parity_across_backends(layer_sizes):
    numpy_net = MLP(layer_sizes=layer_sizes, seed=0)
    torch_net = TorchMLP(layer_sizes=layer_sizes, seed=0)
    numpy_data = numpy_net.to_dict()
    torch_data = torch_net.to_dict()

    assert numpy_data["layer_sizes"] == torch_data["layer_sizes"] == layer_sizes
    assert len(numpy_data["weights"]) == len(torch_data["weights"])
    for nw, tw in zip(numpy_data["weights"], torch_data["weights"]):
        assert len(nw) == len(tw)  # n_in się zgadza
        assert len(nw[0]) == len(tw[0])  # n_out się zgadza -- złapie brakujące .T
    for nb, tb in zip(numpy_data["biases"], torch_data["biases"]):
        assert len(nb) == len(tb)
    # Celowo NIE porównujemy wartości - różne silniki RNG dają różne liczby
    # nawet przy tym samym seedzie, i to jest oczekiwane.
