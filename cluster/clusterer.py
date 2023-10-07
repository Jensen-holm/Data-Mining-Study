from dataclasses import dataclass
from typing import Callable


@dataclass
class Clusterer:
    cluster_func: Callable
    plt_data = None

    def eval(
        self,
        pred_labels,
        true_labels,
    ) -> None:
        ...

    def set_plot_data(self, plt_data):
        self.plt_data = plt_data

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)
