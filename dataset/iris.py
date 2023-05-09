from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def iris():
    """
    returns a tuple of numpy arrays containing the
    iris dataset split into training and testing sets
    after being normalized and one-hot encoded 
    """
    iris = load_iris()
    scaler = StandardScaler()
    x = scaler.fit_transform(iris.data)
    y = OneHotEncoder().fit_transform(iris.target.reshape(-1, 1)).toarray()
    return x, y
