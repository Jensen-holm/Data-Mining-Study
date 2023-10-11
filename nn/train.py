from sklearn.model_selection import train_test_split
from typing import Callable
from nn.nn import NN
import numpy as np


def init_weights_biases(nn: NN) -> None:
    bh = np.zeros((1, nn.hidden_size))
    bo = np.zeros((1, 1))
    wh = np.random.randn(nn.input_size, nn.hidden_size) * \
        np.sqrt(2 / nn.input_size)
    wo = np.random.randn(nn.hidden_size, 1) * np.sqrt(2 / nn.hidden_size)
    return wh, wo, bh, bo


def train(nn: NN) -> dict:
    wh, wo, bh, bo = init_weights_biases(nn=nn)
    X_train, X_test, y_train, y_test = train_test_split(
        nn.X,
        nn.y,
        test_size=nn.test_size,
    )

    mse: float = 0.0
    loss_hist: list[float] = []
    for _ in range(nn.epochs):
        # compute hidden output
        hidden_output = compute_node(
            data=X_train.to_numpy(),
            weights=wh,
            biases=bh,
            func=nn.func,
        )

        # compute output layer
        y_hat = compute_node(
            data=hidden_output,
            weights=wo,
            biases=bo,
            func=nn.func,
        )
        # compute error & store it
        error = y_hat - y_train
        mse = mean_squared_error(y_train, y_hat)
        loss_hist.append(mse)

        # update weights & biases using gradient descent after
        # computing derivatives.
        wh -= (nn.learning_rate * hidden_weight_prime(X_train, error))
        wo -= (nn.learning_rate * output_weight_prime(hidden_output, error))
        bh -= (nn.learning_rate * hidden_bias_prime(error))
        bo -= (nn.learning_rate * output_bias_prime(error))
    return {
        "mse": mse,
        "loss_hist": loss_hist,
    }


def compute_node(data: np.array, weights: np.array, biases: np.array, func: Callable) -> np.array:
    return func(np.dot(data, weights) + biases)


def mean_squared_error(y: np.array, y_hat: np.array) -> np.array:
    return np.mean((y - y_hat) ** 2)


def hidden_weight_prime(data, error):
    return np.dot(data.T, error)


def output_weight_prime(hidden_output, error):
    return np.dot(hidden_output.T, error)


def hidden_bias_prime(error):
    return np.sum(error, axis=0)


def output_bias_prime(error):
    return np.sum(error, axis=0)
