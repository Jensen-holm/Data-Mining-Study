import requests
import json

ENDPOINT: str = "http://127.0.0.1:5000/"

request_params = {
    "algorithm": "kmeans-clustering",
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
