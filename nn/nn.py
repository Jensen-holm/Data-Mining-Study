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
    ):
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.activation = activation
        self.features = features
        self.target = target
        self.data = data

        self.input_size = len(features)

        self.wh: np.array = None
        self.wo: np.array = None
        self.bh: np.array = None
        self.bo: np.array = None
        self.func_prime: Callable = None
        self.func: Callable = None
        self.df: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.y: pd.DataFrame = None

    def set_df(self, df: pd.DataFrame) -> None:
        assert isinstance(df, pd.DataFrame)
        self.df = df
        self.X = df[self.features]
        self.y = df[self.target]

    def set_func(self, f: Callable) -> None:
        assert isinstance(f, Callable)
        self.func = f

    def set_func_prime(self, f: Callable) -> None:
        assert isinstance(f, Callable)
        self.func_prime = f

    def set_bh(self, bh: np.array) -> None:
        self.bh = bh

    def set_wh(self, wh: np.array) -> None:
        self.wh = wh

    def set_bo(self, bo: np.array) -> None:
        self.bo = bo

    def set_wo(self, wo: np.array) -> None:
        self.wo = wo

    @classmethod
    def from_dict(cls, dct):
        """ Creates an instance of NN given a dictionary
        we can use this to make sure that the arguments are right
        """
        return cls(**dct)
