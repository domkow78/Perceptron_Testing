"""Flask: serwuje API sieci neuronowych oraz statyczny frontend."""
from pathlib import Path

from flask import Flask, send_from_directory

from api.perceptron_routes import bp as perceptron_bp

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.register_blueprint(perceptron_bp)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
