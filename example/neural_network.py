import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json


ENDPOINT: str = "http://127.0.0.1:5000/"

request_params = {
    "algorithm": "neural-network",
    "arguments": {
        "epochs": 100,
        "activation_func": "tanh",
        "hidden_size": 8,
        "learning_rate": 0.01
    }
}

headers = {
    "Content-Type": "application/json",
}

r = requests.post(
    ENDPOINT,
    headers=headers,
    data=json.dumps(request_params),
)

model = r.json()


def plot():
    sns.set()
    plt.plot(model["loss_history"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.show()


if __name__ == "__main__":
    plot()
