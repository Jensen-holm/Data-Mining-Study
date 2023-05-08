from dataclasses import dataclass
import numpy as np

from cluster.clusterer import Clusterer


@dataclass
class Kmedoids(Clusterer):
    k: int

    def build(self, X_train: np.array):
        ...

    def label():
        ...

    def main():
        ...
