from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Clusterer:
    cluster_func: Callable
    options: dict

    def eval(
        self,
        pred_labels: np.array,
        true_labels: np.array,
    ) -> None:
        ...

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return {
            "cluster_method": self.cluster_func.__name__,
            "options": self.options,
        }
