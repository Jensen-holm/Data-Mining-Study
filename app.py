from flask import Flask, request, jsonify, make_response, render_template

from dataset.random import random_dataset
from opts import options

app = Flask(
    __name__,
    template_folder="templates",
)


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

    params = request.json
    error_message = not_valid(params=params)
    if error_message:
        return make_response(error_message, 400)

    # parse arguments
    algorithm = options[params["algorithm"]]
    args = params["arguments"]

    # in the future instead of a random data set
    # we should do a more real one like palmer penguins
    X, y = random_dataset(100, 10)
    model = algorithm(X, y, args)
    return jsonify(model)


if __name__ == "__main__":
    app.run(
        debug=False,
    )
