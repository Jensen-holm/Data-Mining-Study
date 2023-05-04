# Data Mining from scratch backend

Currently living [here](https://data-mining-from-scratch-backend.onrender.com/)

### Example Useage

```python
import requests
import json

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
    "https://data-mining-from-scratch-backend.onrender.com/",
    headers=headers,
    data=json.dumps(request_params),
)

model_data = r.json()
print(model_data)
```
