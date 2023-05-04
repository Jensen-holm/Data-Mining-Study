from sklearn.model_selection import train_test_split
from typing import Callable
import numpy as np

# for determing which clustering funciton to call
from cluster.opts import clustering_methods


def main(
    X: np.array,
    y: np.array,
    args: dict,
):

    cluster_alg: Callable = clustering_methods[args["algorithm"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=8675309,
    )

    return
