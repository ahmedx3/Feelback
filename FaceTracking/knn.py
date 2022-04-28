from FaceTracking.feature_extracton import eigen_faces_features
import numpy as np
import numpy.typing as npt
from typing import List


# TODO:
#  Try multiple approaches for adding a new class:
#    1. Avg distance for classified class
#    2. Min distance for classified class
#    3. Avg distance for nearest class
#    4. Min distance for nearest class
#    5. Prefer newly added class
#    6. Use nearest neighbour instead of K-nearest neighbour when adding a new class
#    7. Use KNN with Kmeans: Use kmeans to cluster new samples, when they are large enough add them to KNN


class KNNIdentification:
    """
    This class use a modified version of K Nearest Neighbours (KNN) Classification Algorithm to use in online
    face recognition and identification.
    This class does not need labels, instead it labels the data automatically.

    Notes:
        At the very beginning, the number of available data may be smaller than k, therefore all points are considered,
        and we will probably have a tie.
        Instead, repeat the points of new classes to make them probable to be chosen.
        Each class will have at least these number of points: `1 + k//2`.
    """

    def __init__(self, n_classes: int = -1, k=5, threshold=2000):
        """

        Args:
            n_classes (int): The initial number of classes to classify data into.
                             Defaults to -1, which means it will be inferred from data in `knn_init`.
            k (int): Number of neighbours to use when classifying. Defaults to 5.
            threshold (float): The maximum euclidian distance at which two points are considered in the same cluster.
                               i.e. if the minimum euclidian distance between point A and average of distances of all
                               near neighbours clusters exceeds threshold, a new class will be created with point A.
        """

        self.n_classes = n_classes
        self.k = k
        self.threshold = threshold
        self.features = None
        self.classes = None

    def knn_init(self, x: np.ndarray) -> npt.NDArray[np.uint16]:
        """
        A modified version of K Nearest Neighbours (KNN) Classification Algorithm to use in online face recognition.

        Args:
            x (np.ndarray): array of shape (n_samples, n_features).
                            The data to classify.
        Returns:
            classes : np.ndarray of shape (n_samples,)
                The `classes[i]` is the class index where `x[i]` belongs to.

        Notes:
            This function should be called once to initialize the dimensionality of features and labels, for updating
            existing classes use `knn` instead.
        """

        if self.n_classes == -1:
            self.n_classes = x.shape[0]

        self.features = np.tile(x, (1 + self.k // 2, 1))
        self.classes = np.tile(np.arange(x.shape[0]), 1 + self.k // 2)

        return self.classes[:x.shape[0]]

    def knn(self, x: np.ndarray) -> npt.NDArray[np.uint16]:
        """
        A modified version of K Nearest Neighbours (KNN) Classification Algorithm to use in online face recognition.

        Args:
            x (np.ndarray): array of shape (n_samples, n_features).
                            The data to classify.
        Returns:
            classes : np.ndarray of shape (n_samples,)
                The `classes[i]` is the class index where `x[i]` belongs to.

        Notes:
            At the very beginning, the number of available data may be smaller than k, therefore all points are
            considered, and we will probably have a tie.
            Instead, repeat the points of new classes to make them probable to be chosen.
            Each class will have at least these number of points: `1 + k//2`.
        """

        num_of_test_faces = x.shape[0]
        classes = np.zeros(num_of_test_faces, dtype=np.uint16)

        for i, row in enumerate(x):
            distances = np.linalg.norm(self.features - row, axis=1)
            min_k_distances_classes = np.argsort(distances)[:self.k]
            nearest_k_classes = self.classes[min_k_distances_classes]
            label = np.bincount(nearest_k_classes).argmax()  # get the most frequent class
            classes[i] = label

            min_k_distances = distances[min_k_distances_classes]
            avg_distance_to_classified_class = np.mean(min_k_distances[nearest_k_classes == label])

            # Create a new class if the current row is too far from existing classes
            if avg_distance_to_classified_class > self.threshold:
                classes[i] = self.n_classes
                classes = np.append(classes, np.repeat(self.n_classes, self.k // 2), axis=0)
                x = np.append(x, np.tile(row, (self.k // 2, 1)), axis=0)
                self.n_classes += 1

        self.features = np.append(self.features, x, axis=0)
        self.classes = np.append(self.classes, classes, axis=0)

        return classes[:num_of_test_faces]

    def get_ids(self, faces: List[np.ndarray]) -> npt.NDArray[np.uint16]:
        eigen_faces = eigen_faces_features(faces)
        return self.knn_init(eigen_faces) if self.classes is None else self.knn(eigen_faces)

    def get_outliers_ids(self) -> npt.NDArray[np.int]:
        """
        Returns:
            outliers_ids: np.ndarray which contains ids of outlier classes.
        """

        freq = np.bincount(self.classes)
        # outliers_ids = np.argwhere(np.mean(freq) - freq > np.std(freq))
        # outliers_ids = np.argwhere(freq < np.percentile(freq, 25))
        max_freq = freq.max()
        outliers_ids = np.argwhere((max_freq - freq) / max_freq > 0.75)
        return outliers_ids.ravel()

