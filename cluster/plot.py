import io
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns


matplotlib.use("Agg")
sns.set()

def plot(clusterer, X) -> None:
    cluster_data = clusterer.to_dict(X)["clusters"]
    # plot the clusters and data points
    fig, ax = plt.subplots(figsize=(8, 6))
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
            color="red",
        )
    ax.legend()
    ax.set_title("K-means Clustering")
    ax.set_ylabel("Normalized Petal Length (cm)")
    ax.set_xlabel("Normalized Petal Length (cm)")
    clusterer.plot = plt_bytes(fig)


def plt_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
