import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_embeddings_clusters(
    embeddings: np.ndarray,
    clusters: List[List[int]],
    title: str = "Scatter Plot",
    save_path: str = None,
) -> None:
    """
    Creates a scatter plot of the embeddings.

    Parameters:
    embeddings (np.ndarray): The embeddings to plot.
    clusters (List[List[int]]): List of clusters, each containing indices of the embeddings.
    title (str): The title of the plot. Defaults to "Scatter Plot".
    """
    num_dimensions = embeddings.shape[1]

    if num_dimensions == 2:
        plt.figure(figsize=(10, 8))
        for cluster_id, cluster in enumerate(clusters):
            cluster_embeddings = embeddings[cluster]
            plt.scatter(
                cluster_embeddings[:, 0],
                cluster_embeddings[:, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
            )
        plt.legend()
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    elif num_dimensions == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        for cluster_id, cluster in enumerate(clusters):
            cluster_embeddings = embeddings[cluster]
            ax.scatter(
                cluster_embeddings[:, 0],
                cluster_embeddings[:, 1],
                cluster_embeddings[:, 2],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
            )
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    else:
        print("Warning: Only 2D and 3D embeddings can be plotted.")
