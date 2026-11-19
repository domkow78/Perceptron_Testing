"""Bezstanowe API dla generycznej sieci wielowarstwowej (ten sam wzorzec co perceptron)."""
from flask import Blueprint, jsonify, request

from nn.mlp import MLP

bp = Blueprint("mlp_api", __name__, url_prefix="/api/mlp")


def _build_network(data: dict, with_learning_rate: bool = False) -> MLP:
    kwargs = {"learning_rate": float(data.get("learning_rate", 0.1))} if with_learning_rate else {}
    return MLP.from_dict(
        {
            "layer_sizes": data["architecture"],
            "hidden_activation": data.get("hidden_activation", "sigmoid"),
            "weights": data["weights"],
            "biases": data["biases"],
        },
        **kwargs,
    )


@bp.post("/init")
def init_route():
    data = request.get_json(force=True)
    net = MLP(
        layer_sizes=data["architecture"],
        hidden_activation=data.get("hidden_activation", "sigmoid"),
        seed=data.get("seed"),
    )
    return jsonify(net.to_dict())


@bp.post("/step")
def step_route():
    data = request.get_json(force=True)
    net = _build_network(data, with_learning_rate=True)
    dataset = data["dataset"]
    result = net.train_epoch(dataset["points"], dataset["labels"])
    return jsonify({**net.to_dict(), **result})


@bp.post("/predict")
def predict_route():
    data = request.get_json(force=True)
    net = _build_network(data)
    outputs = net.predict(data["points"])
    return jsonify({"outputs": outputs.reshape(-1).tolist()})


@bp.post("/activations")
def activations_route():
    data = request.get_json(force=True)
    net = _build_network(data)
    activations = net.forward([data["point"]])
    return jsonify({"layers": [a.reshape(-1).tolist() for a in activations]})
