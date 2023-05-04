from sklearn.model_selection import train_test_split
import numpy as np

from neural_network.opts import activation
from neural_network.backprop import bp


def init(
    X: np.array,
    hidden_size: int
) -> dict:
    """
    returns a dictionary containing randomly initialized
    weights and biases to start off the neural_network
    """
    return {
        "w1": np.random.randn(X.shape[1], hidden_size),
        "b1": np.zeros((1, hidden_size)),
        "w2": np.random.randn(hidden_size, 1),
        "b2": np.zeros((1, 1)),
    }


def main( 
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    args,
) -> None:
    wb = init(X_train, args["hidden_size"])
    act = activation[args["activation_func"]]
    args["activation_func"] = act["main"]
    args["func_prime"] = act["prime"]
    model = bp(X_train, y_train, wb, args)

    # evaluate the model and return final results
    model.eval(
        X_test=X_test,
        y_test=y_test,
    )
    return model.to_dict()
