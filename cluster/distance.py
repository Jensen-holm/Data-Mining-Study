import numpy as np


def euclidean(
    point: np.array,
    data: np.array,
) -> np.array:
    """
    Computed the euclidean distance
    between a point and the rest 
    of the dataset
    point dims: (m,)
    data dims: (n, m)
    output dims: (n,)
    """
    return np.sqrt(np.sum((point - data)**2), aixs=1)
