import numpy as np


class PCA:
    """
    Principal component analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional
    space.

    Notes:
        The input data is centered but not scaled for each feature before applying the transformation.
    """

    def __init__(self, n: int):
        """
        Principal component analysis (PCA).

        Args:
            n (int): Number of principal components to keep.
        """
        self.n: int = n
        self.mean = None
        self.principle_components = None

    def fit(self, x: np.ndarray):
        """
        Fit the model with x.

        Args:
            x (np.ndarray): Training data array of shape (n_samples, n_features)

        Returns: None
        """

        self.mean = np.mean(x, axis=0)
        # Center data around origin by convert it to zero mean
        z = x - self.mean

        # use this formula instead of `np.cov(z.T)` because it uses much less memory (about half)
        covariance_matrix = (z.T @ z) / z.shape[0]
        del z

        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        del covariance_matrix
        
        largest_n_eigen_values_indices = eigen_values.argsort()[-self.n:][::-1]
        del eigen_values

        self.principle_components = eigen_vectors[:, largest_n_eigen_values_indices].T

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Principal Component analysis dimensionality reduction to X.

        x is projected on the first n principal components.

        Args:
            x (np.ndarray): array of shape (n_samples, n_features).

        Returns:
            z (np.ndarray): array of shape (n_samples, n)
                            Projection of x on the first n principal components,
                            where `n` is the number of the principal components.
        """

        z = x - self.mean
        return z @ self.principle_components.T

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Try to restore data back to its original space.
        i.e. return an input `x_original` whose `pca.transform` would be x.

        Args:
            x (np.ndarray): array of shape (n_samples, n), where `n` is the number of the principal components.
                            This is the output of `pca.transform()`

        Returns:
            x_original array of shape (n_samples, n_features)
        """

        return (x @ self.principle_components) + self.mean

