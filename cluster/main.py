import numpy as np

from cluster.clusterer import Clusterer
from cluster.opts import clustering_methods
from cluster.plot import plot


def main(
    X: np.array,
    y: np.array,
    clusterer: str,
    args: dict,
) -> dict:
    cluster_alg: Clusterer = clustering_methods[clusterer]

    args.update({"cluster_func": cluster_alg})
    alg = cluster_alg.from_dict(args)

    alg.build(X)
    plot(clusterer=alg, X=X)
    return alg.to_dict(X)
