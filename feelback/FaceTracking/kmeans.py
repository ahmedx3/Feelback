# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

# from ..utils.img_utils import BoundingBox
from ..utils import verbose
from .feature_extracton import eigen_faces_features
import numpy as np
import numpy.typing as npt
from typing import List


class KmeansIdentification:
    """
    This class use a modified version of K-means Algorithm to use in online face recognition and identification
    """

    def __init__(self, k: int = -1, iterations=2000, tolerance=1e-4, threshold=2000, learning_rate=1):
        """

        Args:
            k (int): The initial number of clusters to partition data into.
                     Defaults to -1, which means it will be inferred from data in `kmeans_init`.
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
        self.cluster_count = None
        self.max_iterations = max(10, iterations)
        self.tolerance = tolerance
        self.threshold = threshold
        self.learning_rate = learning_rate

    def kmeans_init(self, x: np.ndarray) -> npt.NDArray[np.uint16]:
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

        if self.k == -1:
            self.k = x.shape[0]

        clusters = np.zeros(x.shape[0], dtype=np.uint16)

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
        self.cluster_count = np.ones(centroids.shape[0], dtype=np.uint16)

        verbose.print(f"kmeans iterations: {self.max_iterations - iterations}", level=verbose.Level.TRACE)
        verbose.print(f"kmeans centroids: {self.centroids}", level=verbose.Level.TRACE)

        return clusters

    def kmeans_dynamic_update(self, x: np.ndarray) -> npt.NDArray[np.uint16]:
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

        clusters = np.zeros(x.shape[0], dtype=np.uint16)
        centroids = self.centroids
        cluster_count = np.zeros(centroids.shape[0], dtype=np.uint16)

        iterations = self.max_iterations

        while iterations > 0:
            for i, row in enumerate(x):
                # The cluster of the ith sample point is the closest one to this sample point
                distances = np.linalg.norm(centroids - row, axis=1)
                cluster_id = np.argmin(distances)
                clusters[i] = cluster_id
                cluster_count[cluster_id] += 1
                min_distance = distances[clusters[i]]

                # Create a new cluster if the current row is too far from existing clusters
                if min_distance > self.threshold:
                    centroids = np.append(centroids, [row], axis=0)
                    cluster_count = np.pad(cluster_count, (0, 1), constant_values=1)
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

        new_clusters_count = cluster_count.shape[0] - self.cluster_count.shape[0]
        # Extend self.cluster_count according to added clusters
        self.cluster_count = np.append(self.cluster_count, np.zeros(new_clusters_count, np.uint16), axis=0)
        # increase count for clusters that appeared in this frame
        self.cluster_count[cluster_count != 0] += 1

        verbose.print(f"kmeans iterations: {self.max_iterations - iterations}", level=verbose.Level.TRACE)
        verbose.print(f"kmeans centroids: {self.centroids}", level=verbose.Level.TRACE)

        return clusters

    def get_ids(self, faces: List[np.ndarray]) -> npt.NDArray[np.uint16]:
        eigen_faces = eigen_faces_features(faces)
        return self.kmeans_init(eigen_faces) if self.centroids is None else self.kmeans_dynamic_update(eigen_faces)
