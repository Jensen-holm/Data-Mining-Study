import numpy as np

relu = lambda x: np.maximum(x, 0)
relu_prime = lambda x: np.where(x > 0, 1, 0)

tanh = lambda x: np.tanh(x)
tanh_prime = lambda x: 1 - tanh(x) ** 2

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
sigmoid_prime = lambda x: sigmoid(x) / 1.0 - sigmoid(x)
