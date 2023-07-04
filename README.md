# Data Mining from scratch backend

Currently living [here](https://data-mining-from-scratch-backend.onrender.com/) <br>
Since the API is hosted using render's free tier, <br>
every time 15 minutes goes by it gets shut down. <br>
If a request is made while it is shut down, the web service <br>
has to spin back up again which takes roughly 1 minute <br>

### Example Useage

```python
import requests
import json

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
    "https://data-mining-from-scratch-backend.onrender.com/neural-network",
    headers=headers,
    data=json.dumps(request_params),
)

model_data = r.json()
print(model_data)
```

### Parameter Options

- End Points: <br>

  -`"neural-network"` <br>

  - `"kmeans-clustering"` <br> -`"kmedoid-clustering"` <br> -`"heirarchical-clustering"` <br>

- Algorithm Specific Arguments

  - neural-network
    - epochs: any integer
    - activation_func: tanh, sigmoid, or relu
    - hidden_size: must be an even integer
    - learning_rate: any floating point number
