from flask import Flask, request, jsonify, make_response

from dataset.random import random_dataset
from opts import options

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


def is_valid(params: dict):
    """
    is_valid() simply checks if
    an incoming response is valid
    for this api or not. Key to
    avoiding errors
    """
    if not request.json():
        return False

    if "Algorithm" not in params:
        return False

    if params["Algorithm"] not in options:
        return False
    return True


@app.route("/", methods=["GET"])
def index():

    params = request.json()
    if not is_valid(params=params):
        return make_response(400, "bad request")

    # parse arguments
    algorithm = options[params["Algorithm"]]
    args = options[params["Arguments"]]

    X, y = random_dataset()
    results = algorithm.main(
        X=X,
        y=y,
        args=args,
    )

    return jsonify(results)


if __name__ == '__main__':
    app.run(
        debug=True,
    )
