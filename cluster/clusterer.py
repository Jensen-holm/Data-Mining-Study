from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Clusterer:
    cluster_func: Callable
    plot = None

    def eval(
        self,
        pred_labels: np.array,
        true_labels: np.array,
    ) -> None:
        ...

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)
