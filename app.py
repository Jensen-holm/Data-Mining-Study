from flask import Flask, request, jsonify, Response
from nn.nn import NN
from nn import train as train_nn
from nn import activation
import pandas as pd
import io

app = Flask(__name__)


@app.route("/neural-network", methods=["POST"])
def neural_net():
    args = request.json

    try:
        net = NN.from_dict(args)
    except Exception as e:
        return Response(
            response=f"issue with request args: {e}",
            status=400,
        )

    try:
        df = pd.read_csv(io.StringIO(net.data))
        net.set_df(df=df)
    except Exception as e:
        return Response(
            response=f"error reading csv data: {e}",
            status=400,
        )

    try:
        activation.get_activation(nn=net)
    except Exception:
        return Response(
            response="invalid activation function",
            status=400,
        )

    result = train_nn.train(nn=net)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
