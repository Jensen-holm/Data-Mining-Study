import numpy as np


def random_dataset(rows: int, features: int):
    """
    the random_dataset function is used to 
    generate a random normal distribution of
    data for testing different machine learning
    algorithms specific to this project
    """
    rng = np.random.default_rng()
    X = rng.normal(size=(rows, features))
    y = rng.integers(5, size=(rows, 1))
    return X, y
