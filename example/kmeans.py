
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns


ENDPOINT: str = "http://127.0.0.1:5000/"

request_params = {
    "arguments": {
        "clusterer": "kmeans-clustering",
        "k": 3,
        "max_iter": 100,
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
