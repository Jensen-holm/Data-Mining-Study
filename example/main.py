import requests

with open("mushrooms.csv", "rb") as csv:
    data = csv.read()

# class,cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat

ARGS = {
    "epochs": 1_000,
    "hidden_size": 8,
    "learning_rate": 0.0001,
    "test_size": 0.1,
    "activation": "relu",
    "features": [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
    ],
    "target": "class",
    "data": data.decode("utf-8"),
}

if __name__ == "__main__":
    r = requests.post(
        "http://127.0.0.1:5000/neural-network",
        json=ARGS,  # Send the data as a JSON object
    )

    print(r.text)
