import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plt_id import generate_image_key
import os


matplotlib.use("Agg")
sns.set()

UPLOAD_FOLDER = os.getcwd() + "/plots"


def plot(clusterer, X) -> None:
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

    image_key = generate_image_key()

    plot_filename = os.path.join(UPLOAD_FOLDER, f"{image_key}.png")
    fig.savefig(plot_filename, format="png")
    plt.close(fig)

    clusterer.plot_key = image_key
