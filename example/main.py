import requests

with open("iris.csv", "rb") as csv:
    iris_data = csv.read()

ARGS = {
    "epochs": 10000,
    "hidden_size": 8,
    "learning_rate": 0.0001,
    "test_size": 0.1,
    "activation": "relu",
    "features": ["sepal width", "sepal length", "petal width", "petal length"],
    "target": "species",
    "data": iris_data.decode("utf-8"),
}

if __name__ == "__main__":
    r = requests.post(
        "http://127.0.0.1:5000/neural-network",
        json=ARGS,  # Send the data as a JSON object
    )

    print(r.text)
