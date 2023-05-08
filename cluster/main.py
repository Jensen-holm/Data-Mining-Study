import numpy as np

from cluster.clusterer import Clusterer
# for determing which clustering funciton to call
from cluster.opts import clustering_methods


def main(
    X: np.array,
    y: np.array,
    args: dict,
) -> dict:
    cluster_func = args.pop("algorithm")
    cluster_alg: Clusterer = clustering_methods[cluster_func]

    cluster_args: dict = {"cluster_func": cluster_func} | args
    alg = cluster_alg.from_dict(cluster_args)

    alg.build(X)
    return alg.to_dict()
