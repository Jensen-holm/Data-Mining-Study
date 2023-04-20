import numpy as np
from typing import Callable


class Model:
    def __init__(self, final_wb: dict[str, np.array], activation_func: Callable):
        self.func = activation_func
        self.final_wb = final_wb
        self.w1 = final_wb["W1"]
        self.w2 = final_wb["W2"]
        self.b1 = final_wb["b1"]
        self.b2 = final_wb["b2"]

    def predict(self, x: np.array) -> np.array:
        n1 = self.compute_node(x, self.w1, self.b1, self.func)
        return self.compute_node(n1, self.w2, self.b2, self.func)

    @staticmethod
    def compute_node(arr, w, b, func):
        return func(np.dot(arr, w) + b)
