import numpy as np


def random_dataset():
    """
    initializes a training and
    a testing dataset in the form
    of numpy arrays
    """
    np.random.seed(8675309)
    return (
        np.random.randn(10000, 10),
        np.random.randint(5, size=(10000, 1)),
    )


def main(method: str, X: np.array, y: np.array):
    pass



if __name__ == "__main__":
    method = input("\nChoose a method to test: ").lower()

    X, y = random_dataset()
    main(
        method=method, 
        X=X, 
        y=y,
    )

