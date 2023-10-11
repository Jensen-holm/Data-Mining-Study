import requests

with open("iris.csv", "rb") as csv:
    iris_data = csv.read()

ARGS = {
    "epochs": 100,
    "hidden_size": 12,
    "learning_rate": 0.01,
    "activation": "tanh",
    "features": ["sepal width", "sepal length", "petal width", "petal length"],
    "target": "species",
    "data": iris_data.decode('utf-8'),
}

r = requests.post(
    "http://127.0.0.1:5000/neural-network",
    json=ARGS,
)

if __name__ == "__main__":
    print(r.json())