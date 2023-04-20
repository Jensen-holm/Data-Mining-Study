import numpy as np

from neural_network.opts import activation


def bp(X_train: np.array, y_train: np.array, wb: dict, args: dict):
    epochs = args["epochs"]
    func = activation[args["activation_func"]]["main"]
    func_prime = activation[args["activation_func"]]["prime"]
    w1, w2 = wb["W1"], wb["W2"]
    b1, b2 = wb["b1"], wb["b2"]
    lr = args["learning_rate"]

    for e in range(epochs):
        # forward prop
        node1 = compute_node(X_train, w1, b1, func)
        y_hat = compute_node(node1, w2, b2, func)
        error = y_hat - y_train

        # backprop
        # right now this is just the weights,
        # we should also update the biases
        dw2 = np.dot(
            node1.T,
            error * func_prime(y_hat),
        )
        dw1 = np.dot(
            X_train.T,
            np.dot(error * func_prime(y_hat), w2.T) * func_prime(node1),
        )

        # update weights & biases
        w1 -= lr * dw1
        w2 -= lr * dw2


def compute_node(X, w, b, func):
    return func(np.dot(X, w) + b)
