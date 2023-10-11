from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score
from typing import Callable
from nn.nn import NN
import numpy as np


def init_weights_biases(nn: NN):
    # np.random.seed(0)
    bh = np.zeros((1, nn.hidden_size))
    bo = np.zeros((1, nn.output_size))
    wh = np.random.randn(nn.input_size, nn.hidden_size) * \
        np.sqrt(2 / nn.input_size)
    wo = np.random.randn(nn.hidden_size, nn.output_size) * \
        np.sqrt(2 / nn.hidden_size)
    return wh, wo, bh, bo


def train(nn: NN) -> dict:
    wh, wo, bh, bo = init_weights_biases(nn=nn)
    X_train, X_test, y_train, y_test = train_test_split(
        nn.X.to_numpy(),
        nn.y_dummy.to_numpy(),
        test_size=nn.test_size,
        # random_state=0,
    )

    ce: float = 0.0
    loss_hist: list[float] = []
    for _ in range(nn.epochs):
        # compute hidden output
        hidden_output = compute_node(
            data=X_train,
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
        mse = mean_squared_error(y=y_train, y_hat=y_hat)
        loss_hist.append(mse)

        # compute derivatives of weights & biases
        # update weights & biases using gradient descent after
        # computing derivatives.
        dwo = nn.learning_rate * output_weight_prime(hidden_output, error)

        # Use NumPy to sum along the first axis (axis=0)
        # and then reshape to match the shape of bo
        dbo = nn.learning_rate * np.sum(output_bias_prime(error), axis=0)

        dhidden = np.dot(error, wo.T) * nn.func_prime(hidden_output)
        dwh = nn.learning_rate * hidden_weight_prime(X_train, dhidden)
        dbh = nn.learning_rate * hidden_bias_prime(dhidden)

        wh -= dwh
        wo -= dwo
        bh -= dbh
        bo -= dbo

    # compute final predictions on data not seen
    hidden_output_test = compute_node(
        data=X_test,
        weights=wh,
        biases=bh,
        func=nn.func,
    )
    y_hat = compute_node(
        data=hidden_output_test,
        weights=wo,
        biases=bo,
        func=nn.func,
    )

    return {
        "log loss": log_loss(y_true=y_test, y_pred=y_hat)
    }


def compute_node(data: np.array, weights: np.array, biases: np.array, func: Callable) -> np.array:
    return func(np.dot(data, weights) + biases)


def mean_squared_error(y: np.array, y_hat: np.array) -> np.array:
    return np.mean((y - y_hat) ** 2)


def hidden_bias_prime(error):
    return np.sum(error, axis=0)


def output_bias_prime(error):
    return np.sum(error, axis=0)


def hidden_weight_prime(data, error):
    return np.dot(data.T, error)


def output_weight_prime(hidden_output, error):
    return np.dot(hidden_output.T, error)
