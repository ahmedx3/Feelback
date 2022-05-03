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
        self.classes_centers = None
        self.classes_count = None

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
        self.classes_centers = x.copy()
        self.classes_count = np.ones(x.shape[0], dtype=np.uint16)

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
            distances_to_class_i = np.linalg.norm(self.features - row, axis=1)
            min_k_distances_classes = np.argsort(distances_to_class_i)[:self.k]
            nearest_k_classes = self.classes[min_k_distances_classes]
            label = np.bincount(nearest_k_classes).argmax()  # get the most frequent class
            classes[i] = label

            # Create a new class if the current row is too far from existing classes
            if np.linalg.norm(self.classes_centers[label] - row) > self.threshold:
                classes, label, x = self.create_new_class(i, row, classes, x)
            else:
                self.incremental_update_class_centers(label, row)

        if np.unique(classes).shape[0] != num_of_test_faces:
            print("[WARNING] There are some conflicts in the classes")
            print(f"[WARNING] Original Classes: {classes}")

            classes, x = self.solve_conflicts(num_of_test_faces, classes, x)

        self.features = np.append(self.features, x, axis=0)
        self.classes = np.append(self.classes, classes, axis=0)

        return classes[:num_of_test_faces]

    def solve_conflicts(self, num_of_test_faces, classes, x):
        """
        Solve conflicts between classes (i.e. when we have two points classified to the same class)

        Args:
            num_of_test_faces (int): number of faces to classify
            classes (np.ndarray): array of shape (n_samples)
                                  the original classes of the test faces.
            x (np.ndarray): array of shape (n_samples, n_features).
                            The original data to classify.

        Returns:
            classes (np.ndarray): array of shape (n_samples)
                                  the new classes of the faces after solving the conflicts.
            x (np.ndarray): array of shape (n_samples, n_features).
                            The new data after solving the conflicts.
        """

        new_classes = np.full_like(classes, fill_value=-1, dtype=np.int16)
        taken_classes = []
        for i in range(num_of_test_faces):
            if i >= self.n_classes:
                index = np.where(new_classes == -1)[0][0]
                new_classes, label, x = self.create_new_class(index, x[index], new_classes, x)

            distances_to_class_i = np.linalg.norm(self.classes_centers[i] - x, axis=1)
            sorted_indices = np.argsort(distances_to_class_i)
            sorted_indices = sorted_indices[np.isin(sorted_indices, taken_classes, invert=True, assume_unique=True)]
            new_classes[sorted_indices[0]] = i
            taken_classes.append(sorted_indices[0])

        # Rollback the effect of misclassified points in the center its original class
        for c in range(num_of_test_faces):
            if new_classes[c] != classes[c]:
                self.incremental_update_class_centers(classes[c], x[c], remove=True)
                self.incremental_update_class_centers(new_classes[c], x[c])

            new_classes[np.where(classes[num_of_test_faces:] == classes[c])[0] + num_of_test_faces] = new_classes[c]

        return new_classes, x

    def create_new_class(self, i, row, classes, x):
        """
        Create a new class for the current row (feature vector).

        Args:
            i (int): index of the current row in the data set.
            row (np.ndarray): feature vector of shape (n_features).
            classes (np.ndarray): labels array of shape (n_samples)
            x (np.ndarray): array of shape (n_samples, n_features).
                            the data set.

        Returns:
            classes (np.ndarray): updated labels array of shape (n_samples)
            label (int): the created label of the new class.
            x (np.ndarray): updated data set of shape (n_samples, n_features).

        Notes:
            The new class will be created by not only adding the current row to it,
            instead, repeat the row `1 + k//2` times to make the class probable to be chosen.
        """

        label = self.n_classes
        classes[i] = label
        classes = np.append(classes, np.repeat(label, self.k // 2), axis=0)
        x = np.append(x, np.tile(row, (self.k // 2, 1)), axis=0)
        self.classes_count = np.append(self.classes_count, [1], axis=0)
        self.classes_centers = np.append(self.classes_centers, [row], axis=0)
        self.n_classes += 1
        return classes, label, x

    def incremental_update_class_centers(self, label, row, remove=False):
        """
        Update the class centers, which is the mean of all points belonging to the class.
        We do this incrementally, so we don't need to recalculate the class centers for each new point.

        Args:
            label (int): the class label.
            row (np.ndarray): array of shape (n_features).
                              the new point to add to the class.
            remove (bool): if True, remove the point from the class.
                           else, add the point to the class, which is the default.
        """

        n = self.classes_count[label]
        if remove:
            self.classes_centers[label] = (self.classes_centers[label] * n - row) / (n - 1)
            self.classes_count[label] -= 1
        else:
            self.classes_centers[label] = (self.classes_centers[label] * n + row) / (n + 1)
            self.classes_count[label] += 1

    def get_ids(self, faces: List[np.ndarray]) -> npt.NDArray[np.uint16]:
        eigen_faces = eigen_faces_features(faces)
        return self.knn_init(eigen_faces) if self.classes is None else self.knn(eigen_faces)

    def get_outliers_ids(self) -> npt.NDArray[np.int]:
        """
        Returns:
            outliers_ids: np.ndarray which contains ids of outlier classes.
        """

        freq = self.classes_count
        # outliers_ids = np.argwhere(np.mean(freq) - freq > np.std(freq))
        # outliers_ids = np.argwhere(freq < np.percentile(freq, 25))
        max_freq = freq.max()
        outliers_ids = np.argwhere((max_freq - freq) / max_freq > 0.75)
        return outliers_ids.ravel()

