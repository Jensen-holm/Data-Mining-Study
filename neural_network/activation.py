import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x: float) -> float:
    return sigmoid(x) / (1.0 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return np.where(x > 0, 1, 0)
