from typing import Callable
from nn.nn import NN
import numpy as np


def get_activation(nn: NN) -> Callable:
    a = nn.activation
    funcs = {
        "relu": relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
    }

    prime_funcs = {
        "sigmoid": sigmoid_prime,
        "tanh": tanh_prime,
        "relu": relu_prime,
    }

    nn.set_func(funcs[a])
    nn.set_func_prime(prime_funcs[a])


def relu(x):
    return np.maximum(0.0, x)


def relu_prime(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2
