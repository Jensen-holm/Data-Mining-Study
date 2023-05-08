from dataclasses import dataclass
import numpy as np

from cluster.clusterer import Clusterer


@dataclass
class Kmedoids(Clusterer):
    k: int

    def main(self, X):
        ...

    def build(self, X: np.array):
        ...
