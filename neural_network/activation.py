import numpy as np



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x: float) -> float:
    return sigmoid(x) / (1.0 - sigmoid())

def relu(x: float) -> float:
    """
    returns the input if > 0
    """
    return max(0.0, x)

def relu_prime(x: float) -> float:
    """
    returns 1 if input is +
    returns 0 if input is -
    """
    return 1 if x > 0 else 0

