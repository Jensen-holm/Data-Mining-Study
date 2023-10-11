from sklearn.model_selection import train_test_split
from typing import Callable
from nn.nn import NN
import pandas as pd
import numpy as np


def init_weights_biases(nn: NN) -> None:
    bh = np.zeros((1, nn.hidden_size))
    bo = np.zeros((1, 1))
    wh = np.random.randn(nn.input_size, nn.hidden_size) * \
        np.sqrt(2 / nn.input_size)
    wo = np.random.randn(nn.hidden_size, 1) * np.sqrt(2 / nn.hidden_size)
    nn.set_bh(bh)
    nn.set_bo(bo)
    nn.set_wh(wh)
    nn.set_wo(wo)


def train(nn: NN) -> dict:
    init_weights_biases(nn=nn)
    X_train, X_test, y_train, y_test = train_test_split(
        nn.X,
        nn.y,
        test_size=nn.test_size,
    )

    for _ in range(nn.epochs):
        # compute hidden output
        hidden_output = compute_node(
            data=X_train.to_numpy(),
            weights=nn.wh,
            biases=nn.bh,
            func=nn.func,
        )

        # compute output layer
        y_hat = compute_node(
            data=hidden_output,
            weights=nn.wo,
            biases=nn.bo,
            func=nn.func,
        )

        mse = mean_squared_error(y_train, y_hat)

    return {"mse": mse}


def compute_node(data: np.array, weights: np.array, biases: np.array, func: Callable) -> np.array:
    return func(np.dot(data, weights) + biases)


def mean_squared_error(y: np.array, y_hat: np.array) -> np.array:
    return np.mean((y - y_hat) ** 2)
