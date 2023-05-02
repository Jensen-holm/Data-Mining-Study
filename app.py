from flask import Flask, render_template, request
import numpy as np

from opts import options

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


def random_dataset(rows: int, features: int):
    """
    Initializes a training and a testing dataset in the form of numpy arrays
    """
    rng = np.random.default_rng()
    X = rng.normal(size=(rows, features))
    y = rng.integers(5, size=(rows, 1))
    return X, y


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/select_algorithm", methods=["GET", "POST"])
def select_algorithm():
    return


@app.route("/process_algorithm", methods=["GET", "POST"])
def process_algorithm():
    alg = request.form.get('model-select')
    func = options[alg]

    # have a form for options based on the algorithm the user chose
    # and set it as the args variable, make a 'go' button for this funcitonality
    # to start the algorithm
    args = request.form.get("params")
    if args:
        # create random numpy array dataset
        X, y = random_dataset(100, 3)
        func(X, y, args)


if __name__ == '__main__':
    app.run(debug=True)
