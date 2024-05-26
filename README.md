# Data Science Library
## Overview
This library is designed to facilitate text embedding, clustering, dimensionality reduction, and visualization of clusters. It can be used both as a command-line interface (CLI) and as a Python module, providing flexibility and ease of use for data scientists and developers.

Features
- `Text Embedding:` Utilizes the sentence_transformers library with the all-MiniLM-v6 model to embed text.

- `Clustering:` Implements Agglomerative Clustering to group similar data points.

- `Dimensionality Reduction:` Supports PCA and UMAP for reducing the dimensionality of data.

- `Visualization:` Creates 2D and 3D scatter plots using matplotlib, color-coded by cluster labels.

## How to Install the Library

Clone the repository:
```bash
git clone https://github.com/abdwalid04/ds_library
```

Navigate to the repository directory:
```bash
cd ds_library
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On macOS and Linux:
```bash
source venv/bin/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

Install the library:
```bash
pip install .
```

## Usage
### As a CLI Tool
The CLI tool provides several options to customize the embedding, clustering, and dimensionality reduction processes. Below are the available arguments and how to use them.

Command-line Arguments

- `--input_file:` Path to the input CSV file (required).
- `--embedding_type:` Type of embedding model. Options: ["sentence_transformer"]. Default: "sentence_transformer".
- `--embedding_model:` Name of the embedding model. Default: "paraphrase-MiniLM-L6-v2".
- `--dim_reduction_algorithm:` Dimensionality reduction algorithm. Options: ["pca", "umap"]. Default: "pca".
- `--num_components:` Number of components for dimensionality reduction. Options: [2, 3]. Default: 2.
- `--clustring_threshold:` Threshold for clustering. Default: 0.7.
- `--min_cluster_size:` Minimum size of a cluster. Default: 3.
- `--clustring_batch_size:` Batch size for clustering. Default: 32.
- `--parallel_clustring:` Flag to indicate if clustering should be done in parallel.
- `--show_clustring_progress_bar:` Flag to indicate if a progress bar should be shown for clustering.
- `--output_file:` Path to save the output plot (required).
- `--output_npy:` Path to save the reduced embeddings as a numpy array (required).


Below is an example command to run the CLI tool:

```bash
ds_library --input_file path/to/your/input.csv --embedding_type sentence_transformer --embedding_model paraphrase-MiniLM-L6-v2 --dim_reduction_algorithm umap --num_components 2 --clustring_threshold 0.7 --min_cluster_size 3 --clustring_batch_size 32 --parallel_clustring --show_clustring_progress_bar --output_file output.png --output_npy output.npy
```
Sample Usage

- `Prepare your CSV file:` Ensure your CSV file has text data in last column. The tool will read this column for text processing. For example:
```csv
Copy code
id,text
1,"This is the first sentence."
2,"This is the second sentence."
3,"This is another sentence."
...
```

- `Run the CLI tool:` Use the command provided above, substituting the example paths and parameters with your actual data and desired settings.

- `Output:` The tool will generate a plot of the clustered data saved to the specified output_file, and the reduced embeddings saved as a numpy array to the specified output_npy.

### As a Python Module
You can also integrate the library into your Python projects. Below is an example of how to use the library:

```python
# Import the necessary components from the library
from ds_library.embedding_service.embedding_service import VectorEmbedService
from ds_library.agglomerative_clustering.agglomerative_clustering import ClusteringService
from ds_library.dimensionality_reduction.dimensionality_reduction import DimensionalityReductionService
from ds_library.visualization_service.visualization_service import VisualizationService

import csv


input_file = "2k_convos.csv"


sentences = []

# Read the CSV file into a list
with open(input_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header line
    for row in reader:
        if row:  # Ensure the row is not empty
            sentences.append(row[-1])

embedding_service = VectorEmbedService(embedder_type="sentence_transformer", model_name ="all-MiniLM-L6-v2")

# Perform clustering
aggl_clustering = AgglomerativeClustering(embedding_service=embedding_service)

clusters = aggl_clustering.cluster(
    sentences=sentences,
    threshold=0.7,
    min_cluster_size=3,
    batch_size=32,
    parallel=False,
    show_progress_bar=True,
)

# Perform dimensionality reduction
dim_reduc = DimensionalityReduction(
    embedding_service=embedding_service,
    algorithm="UMAP",
    num_components=3,
)

reduced_embeddings = dim_reduc.reduce(sentences=sentences, save_path="./output.npy")

# Plot the results
plot_embeddings_clusters(
    embeddings=reduced_embeddings,
    clusters=clusters,
    title=f"UMAP 3D Clustering",
    save_path="./output.png",
)
```