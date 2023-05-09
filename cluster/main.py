import numpy as np

from cluster.clusterer import Clusterer
# for determing which clustering funciton to call
from cluster.opts import clustering_methods
from cluster.plot import plot


def main(
    X: np.array,
    y: np.array,
    args: dict,
) -> dict:
    cluster_func = args.pop("algorithm")
    cluster_alg: Clusterer = clustering_methods[cluster_func]

    args.update({"cluster_func": cluster_func})
    alg = cluster_alg.from_dict(args)

    alg.build(X)
    plot(clusterer=alg, X=X)
    return alg.to_dict(X)
