from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS

from dataset.iris import iris
from opts import options

app = Flask(
    __name__,
    template_folder="templates",
)

CORS(app)


def not_valid(params: dict):
    """
    is_valid() simply checks if
    an incoming response is valid
    for this api or not. Key to
    avoiding errors
    """
    if "algorithm" not in params:
        return "User did not specify the algorithm parameter"

    if params["algorithm"] not in options:
        return f"Invalid algorithm '{params['algorithm']}' is invalid."
    return False


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    error_message = not_valid(params=request.json)
    if error_message:
        return make_response(error_message, 400)

    # parse arguments
    algorithm = options[request.json["algorithm"]]
    args = request.json["arguments"]

    if "cluster" in request.json["algorithm"]:
        args["algorithm"] = request.json["algorithm"]

    # using the iris data set for every algorithm
    X, y = iris()
    result = algorithm(
        X=X,
        y=y,
        args=args,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
