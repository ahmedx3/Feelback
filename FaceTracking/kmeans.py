from typing import Tuple
import numpy as np


def kmeans(x: np.ndarray, k: int, initial_centroids: np.ndarray = None, iterations=2000, tolerance=1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering algorithm

    Args:
        x (np.ndarray): array of shape (n_samples, n_features).
                        The data to cluster.
        k (int): The number of clusters to partition data into.
        initial_centroids (np.ndarray, optional): array of shape (k, n_features).
                                                  It provides the initial clusters' centers.
                                                  Defaults to None which will choose randomized centers.
        iterations (int): Max number of iterations to run the k-means algorithm.
        tolerance (float): Absolute tolerance, used to compare difference in the cluster centers
                           of two consecutive iterations to detect convergence.

    Returns:
        centroids : np.ndarray of shape (k, n_features)
            Centroids found at the last iteration of k-means.

        clusters : np.ndarray of shape (n_samples,)
            The `clusters[i]` is the cluster index where `x[i]` belongs to.
    """

    clusters = np.zeros(x.shape[0])
    centroids = initial_centroids

    if initial_centroids is None:
        # select k random centroids
        initial_indices = np.random.choice(x.shape[0], size=k, replace=False)
        centroids = x[initial_indices, :]

    while iterations > 0:
        for i, row in enumerate(x):
            min_distance = np.inf
            for idx, centroid in enumerate(centroids):
                distance = np.linalg.norm(centroid - row)

                if distance < min_distance:
                    min_distance = distance
                    clusters[i] = idx

        new_centroids = np.array([x[clusters == c].mean(axis=0) for c in np.unique(clusters)])

        # if centroids are same then we have converged
        if np.allclose(new_centroids, centroids, atol=tolerance):
            break

        centroids = new_centroids
        iterations -= 1

    return centroids, clusters

