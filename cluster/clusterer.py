from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Clusterer:
    cluster_func: Callable
    options: dict

    accuracy: float = 0

    @staticmethod
    def label():
        return

    def eval(y_pred, y_true) -> None:
        return

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_dict(self):
        return {
            "cluster_method": self.cluster_func.__name__,
            "options": self.options,
        }
