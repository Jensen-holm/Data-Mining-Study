from dataclasses import dataclass
import numpy as np

from cluster.distance import euclidean
from cluster.clusterer import Clusterer


@dataclass
class Kmeans(Clusterer):
    k: int
    max_iter: int

    def build(
        self,
        X_train: np.array,
    ):
        # Randomly select centroid start points, uniformly distributed across the domain of the dataset
        minimum = np.min(X_train, axis=0)
        maximum = np.max(X_train, axis=0)
        centroids = [np.uniform(minimum, maximum) for _ in range(self.k)]

        # loop through and cluster data
        prev_centroids = 0
        iteration = 0
        while True:
            sorted_pts = [[] for _ in range(self.k)]
            for x in X_train:
                dists = euclidean(x, centroids)

            if not np.not_equal(
                centroids,
                prev_centroids,
            ).any():
                break
            if not iteration < self.k:
                break
            iteration += 1

    def label():
        ...

    def main(self):
        return self.from_dict()
