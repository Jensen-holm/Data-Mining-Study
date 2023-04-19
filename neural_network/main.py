import numpy as np

from neural_network.forwardprop import fp
from neural_network.backprop import bp


def get_args() -> dict:
    """
    returns a dictionary containing
    the arguments to be passed to
    the main function
    """
    return {
        "epochs": int(input("Enter the number of epochs: ")),
        "hidden_size": int(input("Enter the number of hidden nodes: ")),
        "learning_rate": float(input("Enter the learning rate: ")),
        "activation_func": input("Enter the activation function: "),
    }


def init(X: np.array, y: np.array, hidden_size: int) -> dict:
    """
    returns a dictionary containing randomly initialized
    weights and biases to start off the neural_network
    """
    return {
        "W1": np.random.randn(X.shape[1], hidden_size),
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(hidden_size, 1),
        "b2": np.zeros((1, 1)),
    }


def main(
    X: np.array,
    y: np.array,
) -> None:
    args = get_args()
    wb = init(X, y, args["hidden_size"])

    for e in range(args["epochs"]):
        fp()
        bp()

        # update weights and biases

    # print results
