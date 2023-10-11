from sklearn.model_selection import train_test_split
from nn.nn import NN
import pandas as pd
import numpy as np


def init_weights_biases(nn: NN) -> None:
    np.random.seed(88)
    bh = np.zeros((1, 1))
    bo = np.zeros((1, 1))
    wh = np.random.randn(1, nn.input_size) * np.sqrt(2 / nn.input_size)
    wo = np.random.randn(1, nn.hidden_size) * np.sqrt(2 / nn.hidden_size)
    nn.set_bh(bh)
    nn.set_bo(bo)
    nn.set_wh(wh)
    nn.set_wo(wo)


def train(nn: NN) -> dict:
    init_weights_biases(nn=nn)
    X_train, X_test, y_train, y_test = train_test_split(
        nn.X,
        nn.y,
        test_size=nn.test_size,
        random_state=88,
    )

    return {"status": "you made it!"}
