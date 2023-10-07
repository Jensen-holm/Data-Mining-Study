from dataclasses import dataclass, field
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from typing import Callable
import numpy as np
import base64
import binascii
import io

sns.set()
matplotlib.use("Agg")


@dataclass
class NeuralNetwork:
    epochs: int
    learning_rate: float
    activation_func: Callable
    func_prime: Callable
    hidden_size: int
    w1: np.array
    w2: np.array
    b1: np.array
    b2: np.array

    mse: float = 0
    loss_history: list = field(
        default_factory=lambda: [],
    )
    plt_data = None

    def predict(self, x: np.array) -> np.array:
        n1 = self.compute_node(x, self.w1, self.b1, self.activation_func)
        return self.compute_node(n1, self.w2, self.b2, self.activation_func)

    def set_loss_hist(self, loss_hist: list) -> None:
        self.loss_history = loss_hist

    def eval(self, X_test, y_test) -> None:
        self.mse = np.mean((self.predict(X_test) - y_test) ** 2)

    def set_plot_data(self, plot_data):
        self.plt_data = plot_data

    def plot(self):
        sns.set()
        fig, ax = plt.subplots()
        sns.lineplot(
            x=np.arange(len(self.loss_history)),
            y=self.loss_history,
            ax=ax,
        )
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Loss / Epoch")

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt_data = buffer.read()
        plt_data_encoded = binascii.hexlify(plt_data).decode('utf-8')
        self.set_plot_data(plt_data_encoded)
        plt.close()

    @staticmethod
    def compute_node(arr, w, b, func) -> np.array:
        return func(np.dot(arr, w) + b)

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_dict(self) -> dict:
        return {
            # "w1": self.w1.tolist(),
            # "w2": self.w2.tolist(),
            # "b1": self.b1.tolist(),
            # "b2": self.b2.tolist(),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "activation_func": self.activation_func.__name__,
            "func_prime": self.func_prime.__name__,
            "hidden_size": self.hidden_size,
            "mse": self.mse,
            "plt_data": self.plt_data,
        }
