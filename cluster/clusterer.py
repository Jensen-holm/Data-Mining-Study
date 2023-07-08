from dataclasses import dataclass
from typing import Callable


@dataclass
class Clusterer:
    cluster_func: Callable
    plot_key = None

    def eval(
        self,
        pred_labels,
        true_labels,
    ) -> None:
        ...

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)
