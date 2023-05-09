import numpy as np

# right now I am not using this function
# maybe get rid of it or change it to how we 
# use it in our distance calculations

def euclidean(
    point: np.array,
    data: np.array,
):
    """
    Computed the euclidean distance
    between a point and the rest 
    of the dataset
    point dims: (m,)
    data dims: (n, m)
    output dims: (n,)
    """
    return np.sqrt(np.sum((point - data)**2), aixs=1)
