import numpy as np

import neural_network.main as nn


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


if __name__ == "__main__":
    method = input("\nChoose a method to test: ").lower()
    if method != "nn":
        raise ValueError(f"Invalid method '{method}'. Choose 'nn' instead.")

    X, y = random_dataset()
    args = nn.get_args()
    nn.main(
        X=X,
        y=y,
        epochs=args["epochs"],
        hidden_size=args["hidden_size"],
        learning_rate=args["learning_rate"],
        activation_func=args["activation_func"],
    )
