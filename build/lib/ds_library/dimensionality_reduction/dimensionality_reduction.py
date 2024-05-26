import logging
from typing import List, Optional

import numpy as np

import umap

from ds_library.embedding_service import VectorEmbedService

log = logging.getLogger(__name__)


class DimensionalityReduction:
    def __init__(
        self,
        embedding_service: Optional[VectorEmbedService] = None,
        algorithm: str = "pca",
        num_components: int = 2,
    ):
        """
        Initializes the DiversitySampling class with an optional external embedding service.

        Parameters:
            embedding_service (VectorEmbedService, optional): An external service for embedding sentences.
        """
        self.embedding_service = embedding_service
        self.algorithm = algorithm
        self.components = None
        self.variance_share = None
        self.explained_variance_ratio = None
        self.cum_explained_variance = None

        if self.algorithm == "pca":
            self.model = PCA(num_components=num_components)
        elif self.algorithm == "umap":
            self.model = UMAP(
                num_components=num_components
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _embed_sentences(
        self, sentences: List[str], batch_size: int = 32, parallel: bool = False
    ) -> np.ndarray:
        """
        Embeds the given sentences using the internal or external embedding service.

        Parameters:
            sentences (List[str]): A list of sentences to embed.
            batch_size (int, optional): The batch size to use for embedding. Defaults to 32.
            parallel (bool, optional): Flag to indicate if embedding should be done in parallel. Defaults to False.

        Returns:
            np.ndarray: An array of sentence embeddings.
        """
        if self.embedding_service is None:
            raise ValueError("No embedding service provided.")
        return np.array(
            self.embedding_service.embed(
                sentences, batch_size=batch_size, parallel=parallel
            )
        )

    def reduce(self, sentences: List[str], save_path: str = None) -> np.ndarray:
        """
        Fits the dimensionality reduction model to the given sentences.

        Parameters:
            sentences (List[str]): A list of sentences to fit the model to.
        """
        embeddings = self._embed_sentences(sentences)
        reduced_embeddings = self.reduce_with_embeddings(embeddings)
        if save_path:
            np.save(save_path, reduced_embeddings)
        return reduced_embeddings

    def reduce_with_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fits the dimensionality reduction model to the given sentences.

        Parameters:
            embeddings (np.ndarray): An array of sentence embeddings to fit the model to.
        """

        # Fit the model to the embeddings
        self.model.fit(embeddings)

        # Propagate attributes from PCA model to DimensionalityReduction instance
        if self.algorithm == "PCA":
            self.components = self.model.components
            self.variance_share = self.model.variance_share
            self.explained_variance_ratio = self.model.explained_variance_ratio
            self.cum_explained_variance = self.model.cum_explained_variance
        elif self.algorithm == "UMAP":
            self.components = self.model.components

        return self.model.transform(embeddings)


class PCA:
    """
    A PCA model for dimensionality reduction.
    """

    def __init__(self, num_components: int = 2):
        log.info(f"Initializing PCA model with {num_components} components.")
        self.num_components = num_components
        self.components = None
        self.mean = None
        self.scale = None
        self.variance_share = None
        self.explained_variance_ratio = None
        self.cum_explained_variance = None
        self.non_zero_variance_mask = None

    def fit(self, embeddings: np.ndarray):
        """
        Fits the PCA model to the given embeddings.
        Reference: https://github.com/scikit-learn/scikit-learn/blob/5c4aa5d0d/sklearn/decomposition/_pca.py#L518

        Parameters:
            embeddings (np.ndarray): An array of embeddings to fit the model to.

        Returns:
            PCA: The fitted PCA model.
        """
        log.info(f"Fitting PCA model to embeddings of shape: {embeddings.shape}")

        # Check for NaN or Inf values
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain NaN or Inf values.")

        # Remove duplicate rows
        embeddings = np.unique(embeddings, axis=0)

        # Data centering
        self.mean = np.mean(embeddings, axis=0)
        self.scale = np.std(embeddings, axis=0)

        # Create a mask for non-zero variance dimensions
        self.non_zero_variance_mask = self.scale != 0

        # Remove zero variance dimensions and scale the data
        centered_embeddings = (embeddings - self.mean)[:, self.non_zero_variance_mask]
        scale_non_zero = self.scale[self.non_zero_variance_mask]
        centered_embeddings /= scale_non_zero

        # SVD
        try:
            U, S, Vt = np.linalg.svd(centered_embeddings, full_matrices=False)
        except np.linalg.LinAlgError as e:
            log.error("SVD did not converge: %s", e)
            raise

        # Components and explained variance
        self.components = Vt[: self.num_components]

        # The singular values correspond to the variance explained by each component
        total_variance = np.sum(S**2) / (len(embeddings) - 1)
        explained_variance = (S**2) / (len(embeddings) - 1)
        self.explained_variance_ratio = explained_variance / total_variance
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)
        self.variance_share = (
            np.sum(explained_variance[: self.num_components]) / total_variance
        )

        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transforms the given embeddings using the fitted PCA model.

        Parameters:
            embeddings (np.ndarray): An array of embeddings to transform.

        Returns:
            np.ndarray: The transformed embeddings.
        """
        log.info(f"Transforming embeddings of shape: {embeddings.shape}")

        # Data centering
        centered_embeddings = (embeddings - self.mean)[:, self.non_zero_variance_mask]

        # Scale the data
        scale_non_zero = self.scale[self.non_zero_variance_mask]
        centered_embeddings /= scale_non_zero

        return centered_embeddings.dot(self.components.T)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fits the PCA model to the given embeddings and returns the transformed embeddings.

        Parameters:
            embeddings (np.ndarray): An array of embeddings to fit the model to and transform.

        Returns:
            np.ndarray: The transformed embeddings.
        """
        log.info(f"Fitting and transforming embeddings of shape: {embeddings.shape}")
        self.fit(embeddings)
        return self.transform(embeddings)


class UMAP:
    """
    A UMAP model for dimensionality reduction.
    """

    def __init__(
        self, num_components: int = 2
    ):
        log.info(
            f"Initializing UMAP model with {num_components} components."
        )
        self.num_components = num_components
        self.model = umap.UMAP(
            n_components=self.num_components
        )

    def fit(self, embeddings: np.ndarray):
        """
        Fits the UMAP model to the given embeddings.

        Parameters:
            embeddings (np.ndarray): An array of embeddings to fit the model to.

        Returns:
            UMAP: The fitted UMAP model.
        """
        log.info(f"Fitting UMAP model to embeddings of shape: {embeddings.shape}")
        self.model.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transforms the given embeddings using the fitted UMAP model.

        Parameters:
            embeddings (np.ndarray): An array of embeddings to transform.

        Returns:
            np.ndarray: The transformed embeddings.
        """
        log.info(f"Transforming embeddings of shape: {embeddings.shape}")
        return self.model.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fits the UMAP model to the given embeddings and returns the transformed embeddings.

        Parameters:
            embeddings (np.ndarray): An array of embeddings to fit the model to and transform.

        Returns:
            np.ndarray: The transformed embeddings.
        """
        log.info(f"Fitting and transforming embeddings of shape: {embeddings.shape}")
        return self.model.fit_transform(embeddings)
