"""Bezstanowe API: frontend trzyma dataset i wagi, backend liczy kroki/predykcje."""
from flask import Blueprint, jsonify, request

from nn.datasets import get_dataset
from nn.perceptron import Perceptron

bp = Blueprint("perceptron_api", __name__, url_prefix="/api")


@bp.get("/datasets/<name>")
def dataset_route(name):
    try:
        return jsonify(get_dataset(name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@bp.post("/perceptron/predict")
def predict_route():
    data = request.get_json(force=True)
    perceptron = Perceptron.from_dict({"weights": data["weights"], "bias": data["bias"]})
    outputs = [perceptron.predict(point) for point in data["points"]]
    return jsonify({"outputs": outputs})


@bp.post("/perceptron/step")
def step_route():
    data = request.get_json(force=True)
    learning_rate = float(data.get("learning_rate", 0.1))
    perceptron = Perceptron.from_dict(
        {"weights": data["weights"], "bias": data["bias"]}, learning_rate=learning_rate
    )
    dataset = data["dataset"]
    result = perceptron.train_epoch(dataset["points"], dataset["labels"], record_trace=True)
    return jsonify({**perceptron.to_dict(), **result})
