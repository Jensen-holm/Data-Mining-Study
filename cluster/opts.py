from cluster.kmedoids import Kmedoids
from cluster.kmeans import Kmeans


clustering_methods = {
    "kmeans-clustering": Kmeans,
    "kmedoids-clustering": Kmedoids,
}
