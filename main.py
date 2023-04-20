import numpy as np
from opts import options


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


def main():
    method = input("\nChoose a method to test: ").lower()
    try:
        func = options[method]
    except KeyError:
        raise f"Invalid method \"{method}\". Try one of these\n{list(options.keys())}"

    X, y = random_dataset()
    result = func(X, y)


if __name__ == "__main__":
    main()
