"""Bezstanowe API dla generycznej sieci wielowarstwowej (ten sam wzorzec co perceptron).

Backend (NumPy / PyTorch) jest dobierany automatycznie wg rozmiaru sieci
(zobacz nn.backend_select) — frontend nigdy nie musi o tym wiedzieć, ale
może go nadpisać polem "backend" w JSON (przydatne do testów).
"""
from flask import Blueprint, jsonify, request

from nn.backend_select import build_mlp

bp = Blueprint("mlp_api", __name__, url_prefix="/api/mlp")


def _build_network(data: dict, with_learning_rate: bool = False):
    kwargs = {"learning_rate": float(data.get("learning_rate", 0.1))} if with_learning_rate else {}
    return build_mlp(
        layer_sizes=data["architecture"],
        hidden_activation=data.get("hidden_activation", "sigmoid"),
        weights=data["weights"],
        biases=data["biases"],
        backend=data.get("backend"),
        **kwargs,
    )


@bp.post("/init")
def init_route():
    data = request.get_json(force=True)
    net, backend_name = build_mlp(
        layer_sizes=data["architecture"],
        hidden_activation=data.get("hidden_activation", "sigmoid"),
        seed=data.get("seed"),
        backend=data.get("backend"),
    )
    return jsonify({**net.to_dict(), "backend": backend_name})


@bp.post("/step")
def step_route():
    data = request.get_json(force=True)
    net, backend_name = _build_network(data, with_learning_rate=True)
    dataset = data["dataset"]
    result = net.train_epoch(dataset["points"], dataset["labels"])
    return jsonify({**net.to_dict(), **result, "backend": backend_name})


@bp.post("/predict")
def predict_route():
    data = request.get_json(force=True)
    net, _ = _build_network(data)
    outputs = net.predict(data["points"])
    return jsonify({"outputs": outputs.reshape(-1).tolist()})


@bp.post("/activations")
def activations_route():
    data = request.get_json(force=True)
    net, _ = _build_network(data)
    activations = net.forward([data["point"]])
    return jsonify({"layers": [a.reshape(-1).tolist() for a in activations]})
