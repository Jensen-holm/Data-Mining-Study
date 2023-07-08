from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

from dataset.iris import iris
from opts import options

import os

# using the iris data set for every algorithm
X, y = iris()

app = Flask(
    __name__,
    template_folder="templates",
)

CORS(app, origins="*")

UPLOAD_FOLDER = os.getcwd() + "/plots"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/plots/<plt_key>", methods=["GET"])
def get_plot(plt_key):
    filename = f"{plt_key}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.isfile(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        return "Plot not found", 404


@app.route("/neural-network", methods=["POST"])
def neural_network():
    algorithm = options["neural-network"]
    args = request.json["arguments"]

    result = algorithm(
        X=X,
        y=y,
        args=args,
    )
    return jsonify(result)


@app.route("/kmeans-clustering", methods=["POST"])
def kmeans():
    algorithm = options["kmeans-clustering"]
    args = request.json["arguments"]

    result = algorithm(
        X=X,
        y=y,
        clusterer="kmeans-clustering",
        args=args,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)
