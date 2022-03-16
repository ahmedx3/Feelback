import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

from utils.img_utils import BoundingBox
from utils import verbose

import numpy as np
import numpy.typing as npt


class KmeansIdentification:
    """
    This class use a modified version of K-means Algorithm to use in online face recognition and identification
    """

    def __init__(self, k: int, iterations=2000, tolerance=1e-4, threshold=10, learning_rate=0.5):
        """

        Args:
            k (int): The initial number of clusters to partition data into.
            iterations (int): Max number of iterations to run the k-means algorithm.
            tolerance (float): Absolute tolerance, used to compare difference in the cluster centers
                               of two consecutive iterations to detect convergence.
            threshold (float): The maximum euclidian distance at which two points are considered in the same cluster.
                               i.e. if the minimum euclidian distance between point A and centroids of all existing
                               clusters exceeds threshold, a new cluster will be created with centroid is point A.
            learning_rate (float): A float value which controls how much the cluster centroid is updated by the
                                   new sample data.
        """

        self.k = k
        self.centroids = None
        self.max_iterations = max(10, iterations)
        self.tolerance = tolerance
        self.threshold = threshold
        self.learning_rate = learning_rate

    def kmeans_init(self, x: np.ndarray) -> npt.NDArray[int]:
        """
        This is the standard K-means clustering algorithm

        Args:
            x (np.ndarray): array of shape (n_samples, n_features).
                            The data to cluster.
        Returns:
            clusters : np.ndarray of shape (n_samples,)
                The `clusters[i]` is the cluster index where `x[i]` belongs to.

        Notes:
            This function should be called once to initialize the centroid of each cluster, for updating existing
            clusters `kmeans_dynamic_update` should be used instead.
        """

        clusters = np.zeros(x.shape[0], dtype=int)

        # select k random centroids
        np.random.seed(123)
        initial_indices = np.random.choice(x.shape[0], size=self.k, replace=False)
        centroids = x[initial_indices, :]

        iterations = self.max_iterations

        while iterations > 0:
            for i, row in enumerate(x):
                # The cluster of the ith sample point is the closest one to this sample point
                distances = np.linalg.norm(centroids - row, axis=1)
                clusters[i] = np.argmin(distances)

            new_centroids = np.array([x[clusters == c].mean(axis=0) for c in np.unique(clusters)])

            # if centroids are same then we have converged
            if np.allclose(new_centroids, centroids, atol=self.tolerance):
                break

            centroids = new_centroids
            iterations -= 1

        self.centroids = centroids

        verbose.print(f"kmeans iterations: {self.max_iterations - iterations}")
        verbose.print(f"kmeans centroids: {self.centroids}")

        return clusters

    def kmeans_dynamic_update(self, x: np.ndarray) -> npt.NDArray[int]:
        """
        This is a modified version of K-means clustering algorithm.
        It does not use a fixed number of clusters k, instead it creates new clusters if the minimum euclidian distance
        between a point and centroids of all existing clusters exceeds threshold.

        Args:
            x (np.ndarray): array of shape (n_samples, n_features).
                            The data to cluster.
        Returns:
            clusters : np.ndarray of shape (n_samples,)
                The `clusters[i]` is the cluster index where `x[i]` belongs to.
        Notes:
            This function updates the number of existing clusters `k` and the centroid of each cluster.

        Warnings:
            This function should NOT be called with empty centroids, centroids can be initialized by calling
            `kmeans_init`.
        """

        clusters = np.zeros(x.shape[0], dtype=int)
        centroids = self.centroids

        iterations = self.max_iterations

        while iterations > 0:
            for i, row in enumerate(x):
                # The cluster of the ith sample point is the closest one to this sample point
                distances = np.linalg.norm(centroids - row, axis=1)
                clusters[i] = np.argmin(distances)
                min_distance = distances[clusters[i]]

                # Create a new cluster if the current row is too far from existing clusters
                if min_distance > self.threshold:
                    centroids = np.append(centroids, [row], axis=0)
                    clusters[i] = self.k
                    self.k += 1

            new_centroids = np.array(
                [np.vstack((x[clusters == c] * self.learning_rate, centroids[c])).mean(axis=0) for c in range(self.k)])

            # if centroids are same then we have converged
            if np.allclose(new_centroids, centroids, atol=self.tolerance):
                break

            centroids = new_centroids
            iterations -= 1

        self.centroids = centroids

        verbose.print(f"kmeans iterations: {self.max_iterations - iterations}")
        verbose.print(f"kmeans centroids: {self.centroids}")

        return clusters

    def get_ids(self, frame: np.ndarray, faces_positions: npt.NDArray[BoundingBox]) -> npt.NDArray[int]:
        # TODO: Feature Extraction from faces (Eigen-faces + position)
        x = np.array([])
        return self.kmeans_init(x) if self.centroids is None else self.kmeans_dynamic_update(x)
