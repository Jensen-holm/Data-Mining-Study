from sklearn.model_selection import train_test_split
import numpy as np

from cluster.clusterer import Clusterer
# for determing which clustering funciton to call
from cluster.opts import clustering_methods


def main(
    X: np.array,
    y: np.array,
    args: dict,
) -> dict:
    cluster_alg: Clusterer = clustering_methods[args["algorithm"]]

    alg = cluster_alg.from_dict(args)
    alg.build(X)

    return alg.to_dict()
