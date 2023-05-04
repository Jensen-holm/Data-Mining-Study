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
    X: np.array,
    y: np.array,
    args,
) -> None:
    wb = init(X, args["hidden_size"])
    act = activation[args["activation_func"]]
    args["activation_func"] = act["main"]
    args["func_prime"] = act["prime"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=8675309,
    )

    model = bp(X_train, y_train, wb, args)

    # evaluate the model and return final results
    model.eval(
        X_test=X_test,
        y_test=y_test,
    )
    return model.to_dict()
