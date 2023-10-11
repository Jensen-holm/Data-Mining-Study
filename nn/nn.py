from typing import Callable
import pandas as pd
import numpy as np


class NN:
    def __init__(
        self,
        epochs: int,
        hidden_size: int,
        learning_rate: float,
        test_size: float,
        activation: str,
        features: list[str],
        target: str,
        data: str,

        wh: np.array,
        wo: np.array,
        bh: np.array,
        bo: np.array,
    ):
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.activation = activation
        self.features = features
        self.target = target
        self.data = data
        self.wh: np.array = wh
        self.wo: np.array = wo
        self.bh: np.array = bh
        self.bo: np.array = bo

        self.func_prime: Callable = None
        self.func: Callable = None
        self.df: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.y: pd.DataFrame = None

    def read_csv(self) -> dict[str, str]:
        self.df = pd.read_csv(self.data)
        self.X = self.df[self.features]
        self.y = self.df[self.target]

    def set_func(self, f: Callable) -> None:
        assert isinstance(f, Callable)
        self.func = f

    def set_func_prime(self, f: Callable) -> None:
        assert isinstance(f, Callable)
        self.func_prime = f

    @classmethod
    def from_dict(cls, dct):
        """ Creates an instance of NN given a dictionary
        we can use this to make sure that the arguments are right
        """
        return cls(**dct)
