from dataclasses import dataclass
import numpy as np

from cluster.clusterer import Clusterer


@dataclass
class Kmeans(Clusterer):
    k: int
    max_iter: int

    def build(
        self,
        X: np.array,
    ) -> dict[str, np.array]:
        # randomly initialize centroids
        centroids = X[np.random.choice(
            X.shape[0],
            self.k,
            replace=False,
        )]

        # Calculate Euclidean distance between each data point and each centroid
        # then assign each point to its closest cluster
        clusters = self.assign_clusters(X, centroids)
        centroids = self.update_centroids(self.k, X, clusters)

        while True:
            new_clusts = self.assign_clusters(X, centroids)
            if np.array_equal(new_clusts, clusters):
                break
            clusters = new_clusts
            centroids = self.update_centroids(self.k, X, clusters)
        return {
            "clusters": clusters,
            "centroids": centroids,
        }

    @staticmethod
    def assign_clusters(
        X: np.array,
        centroids: np.array,
    ) -> np.array:
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        clusts = np.argmin(distances, axis=0)
        return clusts

    @staticmethod
    def update_centroids(
        k: int,
        X: np.array,
        clusters: np.array,
    ) -> np.array:
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            centroids[i] = X[clusters == i].mean(axis=0)
        return centroids

    def label():
        ...

    def main(self):
        return self.from_dict()
