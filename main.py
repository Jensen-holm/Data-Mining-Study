from opts import options
import numpy as np


def random_dataset(rows: int, features: int):
    """
    Initializes a training and a testing dataset in the form of numpy arrays
    """
    rng = np.random.default_rng()
    X = rng.normal(size=(rows, features))
    y = rng.integers(5, size=(rows, 1))
    return X, y


def main():
    method = input("\nChoose a method to test: ").lower()
    try:
        func = options[method]
    except KeyError:
        raise f"Invalid method \"{method}\". Try one of these\n{list(options.keys())}"

    X, y = random_dataset(rows=10, features=2)
    func(X, y)


if __name__ == "__main__":
    main()
