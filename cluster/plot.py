import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io


matplotlib.use("Agg")
sns.set()


def plot(clusterer, X):
    cluster_data = clusterer.to_dict(X)["clusters"]
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

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer.read()
