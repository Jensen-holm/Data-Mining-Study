from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def iris() -> tuple[np.array]:
    """
    returns a tuple of numpy arrays containing the
    iris dataset split into training and testing sets
    after being normalized and one-hot encoded 
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.3,
        random_state=8675309,
    )
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(
        X_train
    ), scaler.fit_transform(
        X_test
    )

    y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
    return X_train, X_test, y_train, y_test
