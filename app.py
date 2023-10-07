from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from dataset.iris import iris
from opts import options

# using the iris data set for every algorithm
# just for simplicity sake
X, y = iris()

app = Flask(__name__)

CORS(app, origins="*")


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
