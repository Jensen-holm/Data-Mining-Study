import numpy as np
from typing import Callable

from neural_network.neural_network import NeuralNetwork


def fp(
    X_train: np.array,
    y_train: np.array,
    func: Callable,
    w1: np.array,
    w2: np.array,
    b1: np.array,
    b2: np.array,
):
    n1 = compute_node(arr=X_train, w=w1, b=b1, func=func)
    y_hat = compute_node(arr=n1, w=w2, b=b2, func=func)
    return y_hat, n1, (y_hat-y_train)


def bp(
    X_train: np.array,
    y_train: np.array,
    wb: dict,
    args: dict,
) -> NeuralNetwork:
    model = NeuralNetwork.from_dict(args | wb)
    loss_history = []
    for _ in range(model.epochs):
        # forward prop
        y_hat, node1, error = fp(
            X_train=X_train,
            y_train=y_train,
            func=model.activation_func,
            w1=model.w1, w2=model.w2, b1=model.b1, b2=model.b2,
        )
        mean_squared_error = mse(y_train, y_hat)
        loss_history.append(mean_squared_error)

        # backprop
        dw1 = np.dot(
            X_train.T,
            np.dot(error * model.func_prime(y_hat), model.w2.T) *
            model.func_prime(node1),
        )
        dw2 = np.dot(
            node1.T,
            error * model.func_prime(y_hat),
        )
        db2 = np.sum(error * model.func_prime(y_hat), axis=0)
        db1 = np.sum(np.dot(error * model.func_prime(y_hat), model.w2.T)
                     * model.func_prime(node1), axis=0)

        # update weights & biases using gradient descent.
        # this is -= and not += because if the gradient descent
        # is positive, we want to go down.
        model.w1 -= (model.learning_rate * dw1)
        model.w2 -= (model.learning_rate * dw2)
        model.b1 -= (model.learning_rate * db1)
        model.b2 -= (model.learning_rate * db2)

    model.set_loss_hist(loss_hist=loss_history)
    return model


def compute_node(arr, w, b, func):
    """
    Computes nodes during forward prop
    """
    return func(np.dot(arr, w) + b)


def mse(y: np.array, y_hat: np.array):
    return np.mean((y - y_hat) ** 2)
