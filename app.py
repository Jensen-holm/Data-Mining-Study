from flask import Flask, request, jsonify

import pandas as pd
from nn.nn import NN
from nn import train as train_nn

app = Flask(__name__)


@app.route("/neural-network", methods=["POST"])
def neural_net():
    args = request.json

    try:
        net = NN.from_dict(args)
        df = pd.read_csv(args.pop("data"))
    except Exception as e:
        return jsonify({
            "bad request": f"could not read csv data: {e}",
        })

    result = train_nn(nn=net)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
