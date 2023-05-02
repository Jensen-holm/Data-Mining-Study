import numpy as np
from tqdm import tqdm

from neural_network.opts import activation


def get_args(args: dict, wb: dict):
    return (
        args["epochs"],
        args["activation_func"],
        args["func_prime"],
        args["learning_rate"],
        wb["W1"],
        wb["W2"],
        wb["b1"],
        wb["b2"],
    )


def fp(
    X_train: np.array,
    y_train: np.array,
    actiavtion: callable,
    w1: np.array,
    w2: np.array,
    b1: np.array,
    b2: np.array,
):
    n1 = compute_node(arr=X_train, w=w1, b=b1, func=activation)
    y_hat = compute_node(arr=n1, w=w2, b=b2, func=activation)
    return y_hat, n1, (y_hat-y_train)


def bp(
    X_train: np.array,
    y_train: np.array,
    wb: dict,
    args: dict
):
    epochs, func, func_prime, lr, w1, w2, b1, b2 = get_args(args, wb)
    r = {}
    loss_history = []
    for e in tqdm(range(epochs)):
        # forward prop
        y_hat, node1, error = fp(
            X_train=X_train,
            y_train=y_train,
            actiavtion=func,
            w1=w1, w2=w2, b1=b1, b2=b2,
        )
        mean_squared_error = mse(y_train, y_hat)
        loss_history.append(mean_squared_error)

        # backprop
        dw1 = np.dot(
            X_train.T,
            np.dot(error * func_prime(y_hat), w2.T) * func_prime(node1),
        )
        dw2 = np.dot(
            node1.T,
            error * func_prime(y_hat),
        )
        db2 = np.sum(error * func_prime(y_hat), axis=0)
        db1 = np.sum(np.dot(error * func_prime(y_hat), w2.T)
                     * func_prime(node1), axis=0)

        # update weights & biases using gradient descent.
        # this is -= and not += because if the gradient descent
        # is positive, we want to go down.
        w1 -= (lr * dw1)
        w2 -= (lr * dw2)
        b1 -= (lr * db1)
        b2 -= (lr * db2)

        # keeping track of each epochs' numbers
        r[e] = {
            "W1": w1,
            "W2": w2,
            "b1": b1,
            "b2": b2,
            "dw1": dw1,
            "dw2": dw2,
            "db1": db1,
            "db2": db2,
            "error": error,
            "mse": mean_squared_error,
        }
    return r, loss_history


def compute_node(arr, w, b, func):
    """
    Computes nodes during forward prop
    """
    return func(np.dot(arr, w) + b)


def mse(y: np.array, y_hat: np.array):
    return np.mean((y - y_hat) ** 2)
