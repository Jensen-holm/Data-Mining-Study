import requests
import json

ENDPOINT: str = "https://data-mining-from-scratch-backend.onrender.com/"

request_params = {
    "algorithm": "kmeans",
    "arguments": {
        "k": 3,
        "max_iter": 10,
    },
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
    print(r.json())
