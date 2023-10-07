import requests
import json


ENDPOINT: str = "http://127.0.0.1:5000/neural-network"

request_params = {
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

if __name__ == "__main__":
    print(r.json()["plt_data"])
