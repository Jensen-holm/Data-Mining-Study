
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns


ENDPOINT: str = "http://127.0.0.1:5000/"

request_params = {
    "algorithm": "kmeans-clustering",
    "arguments": {
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


def plot():
    cluster_data = r.json()["clusters"]
    # plot the clusters and data points
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set()
    for cluster in cluster_data:
        sns.scatterplot(
            x=[point[0] for point in cluster["points"]],
            y=[point[1] for point in cluster["points"]],
            label=f"Cluster {cluster['cluster_id']}",
            ax=ax,
        )
        ax.scatter(
            x=cluster["centroid"][0],
            y=cluster["centroid"][1],
            marker="x",
            s=100,
            linewidth=2,
            color="black"
        )
    ax.set_title("K-means Clustering")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot()
