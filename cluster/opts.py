from cluster.clusterer import Clusterer
from cluster.kmedoids import Kmedoids
from cluster.kmeans import Kmeans


clustering_methods: dict[str, Clusterer] = {
    "kmeans": Kmeans,
    "kmedoids": Kmedoids,
}
