import pandas as pd


class NN:
    def __init__(
        self,
        epochs: int,
        hidden_size: int,
        learning_rate: float,
        test_size: float,
        activation: str,
        features: list[str],
        target: str,
        data: str,
    ):
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.activation = activation
        self.features = features
        self.target = target
        self.data = data

        self.df: pd.DataFrame = None

    @classmethod
    def from_dict(cls, dct):
        """ Creates an instance of NN given a dictionary
        we can use this to make sure that the arguments are right
        """
        return cls(**dct)
