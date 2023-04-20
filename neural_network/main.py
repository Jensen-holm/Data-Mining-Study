from sklearn.model_selection import train_test_split
from neural_network.backprop import bp
import numpy as np


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
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=8675309
    )

    results = bp(X_train, y_train, wb, args)
