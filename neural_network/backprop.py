import numpy as np

from neural_network.opts import activation


def bp(X_train: np.array, y_train: np.array, wb: dict, args: dict):
    epochs = args["epochs"]
    func = activation[args["activation_func"]]["main"]
    func_prime = activation[args["activation_func"]]["prime"]
    w1, w2 = wb["W1"], wb["W2"]
    b1, b2 = wb["b1"], wb["b2"]
    lr = args["learning_rate"]

    r = {}
    for e in range(epochs):
        # forward prop
        node1 = compute_node(X_train, w1, b1, func)
        y_hat = compute_node(node1, w2, b2, func)
        error = y_hat - y_train

        # backprop
        dw2 = np.dot(
            node1.T,
            error * func_prime(y_hat),
        )
        dw1 = np.dot(
            X_train.T,
            np.dot(error * func_prime(y_hat), w2.T) * func_prime(node1),
        )
        db2 = np.sum(error * func_prime(y_hat), axis=0)
        db1 = np.sum(np.dot(error * func_prime(y_hat), w2.T) * func_prime(node1), axis=0)

        # update weights & biases using gradient descent.
        # this is -= and not += because if the gradient descent
        # is positive, we want to go down.
        w1 -= (lr * dw1)
        w2 -= (lr * dw2)
        b1 -= (lr * db1)
        b2 -= (lr * db2)

        r[e] = {
            "W1": w1,
            "W2": w2,
            "b1": b1,
            "b2": b2,
            "dw1": dw1,
            "dw2": dw2,
            "db1": db1,
            "db2": db2,
        }
    return r


def compute_node(X, w, b, func):
    return func(np.dot(X, w) + b)
