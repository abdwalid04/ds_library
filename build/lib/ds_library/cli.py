import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
from ds_library.embedding_service import VectorEmbedService
from ds_library.agglomerative_clustering import AgglomerativeClustering
from ds_library.dimensionality_reduction import DimensionalityReduction
from ds_library.visualization_service import plot_embeddings_clusters


def main():

    parser = argparse.ArgumentParser(description="Data Science Library CLI")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["sentence_transformer"],
        default="sentence_transformer",
        help="Type of embedding model",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="paraphrase-MiniLM-L6-v2",
        help="Name of the embedding model",
    )
    parser.add_argument(
        "--dim_reduction_algorithm",
        type=str,
        choices=["pca", "umap"],
        default="pca",
        help="Dimensionality reduction algorithm",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of components for dimensionality reduction",
    )
    parser.add_argument(
        "--clustring_threshold",
        type=float,
        default=0.7,
        help="Threshold for clustering",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=3,
        help="Minimum size of a cluster",
    )
    parser.add_argument(
        "--clustring_batch_size",
        type=int,
        default=32,
        help="Batch size for clustering",
    )
    parser.add_argument(
        "--parallel_clustring",
        action="store_true",
        help="Flag to indicate if clustering should be done in parallel",
    )
    parser.add_argument(
        "--show_clustring_progress_bar",
        action="store_true",
        help="Flag to indicate if a progress bar should be shown for clustering",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save the output plot"
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        required=True,
        help="Path to save the reduced embeddings as numpy array",
    )

    args = parser.parse_args()

    sentences = []

    # Read the CSV file into a list
    with open(args.input_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header line
        for row in reader:
            if row:  # Ensure the row is not empty
                sentences.append(row[-1])

    embedding_service = VectorEmbedService(
        embedder_type=args.embedding_type, model_name=args.embedding_model
    )

    # Perform clustering
    aggl_clustering = AgglomerativeClustering(embedding_service=embedding_service)

    clusters = aggl_clustering.cluster(
        sentences=sentences,
        threshold=args.clustring_threshold,
        min_cluster_size=args.min_cluster_size,
        batch_size=args.clustring_batch_size,
        parallel=args.parallel_clustring,
        show_progress_bar=args.show_clustring_progress_bar,
    )

    # Perform dimensionality reduction
    dim_reduc = DimensionalityReduction(
        embedding_service=embedding_service,
        algorithm=args.dim_reduction_algorithm,
        num_components=args.num_components,
    )

    reduced_embeddings = dim_reduc.reduce(
        sentences=sentences, save_path=args.output_npy
    )

    # Plot the results
    plot_embeddings_clusters(
        embeddings=reduced_embeddings,
        clusters=clusters,
        title=f"{args.dim_reduction_algorithm.upper()} {args.num_components}D Clustering",
        save_path=args.output_file,
    )

    print(f"Plot saved to {args.output_file} and embeddings saved to {args.output_npy}")


if __name__ == "__main__":

    main()
