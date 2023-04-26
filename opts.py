from neural_network.main import main as nn
from cluster.main import main as clust

options = {
    "neural-network": nn,
    "kmeans-clustering": clust,
    "kmedoid-clustering": clust,
}
