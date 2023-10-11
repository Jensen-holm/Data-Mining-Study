from sklearn.model_selection import train_test_split
from nn.nn import NN
import pandas as pd
import numpy as np


def train(nn: NN) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        nn.X,
        nn.y,
        test_size=nn.test_size,
        random_state=88,
    )

    return {"status": "you made it!"}
