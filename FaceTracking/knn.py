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

    def __init__(self, n_classes: int = -1, k=5, threshold=2000, conflict_solving_strategy="min_distance"):
        """

        Args:
            n_classes (int): The initial number of classes to classify data into.
                             Defaults to -1, which means it will be inferred from data in `knn_init`.
            k (int): Number of neighbours to use when classifying. Defaults to 5.
            threshold (float): The maximum euclidian distance at which two points are considered in the same cluster.
                               i.e. if the minimum euclidian distance between point A and average of distances of all
                               near neighbours clusters exceeds threshold, a new class will be created with point A.
            conflict_solving_strategy: The strategy to use when there is a conflict between two classes
                                       (i.e. two points have the same label).
                                       Defaults to "min_distance".
                                       Possible values:
                                       - "position": The class with near spatial position will be chosen.
                                       - "min_distance": The class with the minimum distance to the new point is chosen.
        """

        self.n_classes = n_classes
        self.k = k
        self.threshold = threshold
        self.features = None
        self.classes = None
        self.classes_spatial_positions = None
        self.classes_centers = None
        self.classes_count = None
        self.conflict_solving_strategy = conflict_solving_strategy

    def knn_init(self, faces: np.ndarray, faces_positions=None) -> npt.NDArray[np.uint16]:
        """
        A modified version of K Nearest Neighbours (KNN) Classification Algorithm to use in online face recognition.

        Args:
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The faces' data to classify.
            faces_positions (np.ndarray): array of shape (n_samples, 2).
                                          The spatial position of each face in the image.
                                          It is used when conflict_solving_strategy is "position".
                                          If None, the conflict_solving_strategy will fall back to "min_distance".
        Returns:
            classes : np.ndarray of shape (n_samples,)
                The `classes[i]` is the class index where `x[i]` belongs to.

        Notes:
            This function should be called once to initialize the dimensionality of features and labels, for updating
            existing classes use `knn` instead.
        """

        if self.n_classes == -1:
            self.n_classes = faces.shape[0]

        self.features = np.tile(faces, (1 + self.k // 2, 1))
        self.classes = np.tile(np.arange(faces.shape[0]), 1 + self.k // 2)
        self.classes_centers = faces.copy()
        self.classes_spatial_positions = np.copy(faces_positions)
        self.classes_count = np.ones(faces.shape[0], dtype=np.uint16)

        return self.classes[:faces.shape[0]]

    def knn(self, faces: np.ndarray, faces_positions=None) -> npt.NDArray[np.uint16]:
        """
        A modified version of K Nearest Neighbours (KNN) Classification Algorithm to use in online face recognition.

        Args:
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The data to classify.
            faces_positions (np.ndarray): array of shape (n_samples, 2).
                                          The spatial position of each face in the image.
                                          It is used when conflict_solving_strategy is "position".
                                          If None, the conflict_solving_strategy will fall back to "min_distance".
        Returns:
            classes : np.ndarray of shape (n_samples,)
                The `classes[i]` is the class index where `x[i]` belongs to.

        Notes:
            At the very beginning, the number of available data may be smaller than k, therefore all points are
            considered, and we will probably have a tie.
            Instead, repeat the points of new classes to make them probable to be chosen.
            Each class will have at least these number of points: `1 + k//2`.
        """

        num_of_test_faces = faces.shape[0]
        classes = np.zeros(num_of_test_faces, dtype=np.uint16)

        for i, row in enumerate(faces):
            distances_to_class_i = np.linalg.norm(self.features - row, axis=1)
            min_k_distances_classes = np.argsort(distances_to_class_i)[:self.k]
            nearest_k_classes = self.classes[min_k_distances_classes]
            label = np.bincount(nearest_k_classes).argmax()  # get the most frequent class
            classes[i] = label

            # Create a new class if the current row is too far from existing classes
            if np.linalg.norm(self.classes_centers[label] - row) > self.threshold:
                classes, label, faces = self.create_new_class(i, row, classes, faces, faces_positions)
            else:
                self.incremental_update_class_info(label, i, faces, faces_positions)

        if np.unique(classes).shape[0] != num_of_test_faces:
            print("[WARNING] There are some conflicts in the classes")
            print(f"[WARNING] Original Classes: {classes}")

            classes, faces = self.solve_conflicts(num_of_test_faces, classes, faces, faces_positions)

        self.features = np.append(self.features, faces, axis=0)
        self.classes = np.append(self.classes, classes, axis=0)

        return classes[:num_of_test_faces]

    def solve_conflicts(self, num_of_test_faces, classes, faces, faces_positions):
        """
        Solve conflicts between classes (i.e. when we have two points classified to the same class)
        There are two ways to solve the conflicts:
            1. Use the spatial position of the face in the image to resolve the conflict.
            2. Use the distance between the face and the class center to resolve the conflict.

        Args:
            num_of_test_faces (int): number of faces to classify.
            classes (np.ndarray): array of shape (n_samples)
                                  The original classes of the test faces.
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The original data to classify.
            faces_positions (np.ndarray): array of shape (n_samples, 2).
                                          The spatial position of the faces in the image.

        Returns:
            classes (np.ndarray): array of shape (n_samples)
                                  The new classes of the faces after solving the conflicts.
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The new data after solving the conflicts.
        """

        if self.conflict_solving_strategy == "position":
            new_classes, faces = self.solve_conflicts_position(classes, faces, faces_positions)
        else:
            new_classes, faces = self.solve_conflicts_min_distance(num_of_test_faces, classes, faces)

        # Rollback the effect of misclassified points in the center its original class
        for c in range(num_of_test_faces):
            if new_classes[c] != classes[c]:
                self.incremental_update_class_info(classes[c], c, faces, faces_positions, remove=True)
                self.incremental_update_class_info(new_classes[c], c, faces, faces_positions)

            # Update the duplicate elements (which are repeated 1+k/2 times due to newly created classes)
            new_classes[np.where(classes[num_of_test_faces:] == classes[c])[0] + num_of_test_faces] = new_classes[c]

        return new_classes, faces

    def solve_conflicts_min_distance(self, num_of_test_faces, classes, faces):
        """
        Solve conflicts between classes (i.e. when we have two points classified to the same class)
        Using the minimum distance between the point and the class center.

        Args:
            num_of_test_faces (int): number of faces to classify
            classes (np.ndarray): array of shape (n_samples)
                                  the original classes of the test faces.
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The original data to classify.

        Returns:
            classes (np.ndarray): array of shape (n_samples)
                                  the new classes of the faces after solving the conflicts.
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The new data after solving the conflicts.
        """

        new_classes = np.full_like(classes, fill_value=-1, dtype=np.int16)
        taken_classes = []
        for i in range(num_of_test_faces):
            if i >= self.n_classes:
                # Create a new class for the first unclassified face (its class is still -1)
                index = np.where(new_classes == -1)[0][0]
                new_classes, label, faces = self.create_new_class(index, faces[index], new_classes, faces)

            distances_to_class_i = np.linalg.norm(self.classes_centers[i] - faces, axis=1)
            sorted_indices = np.argsort(distances_to_class_i)
            # Remove already taken classes from the candidate classes
            sorted_indices = sorted_indices[np.isin(sorted_indices, taken_classes, invert=True, assume_unique=True)]

            new_classes[sorted_indices[0]] = i
            taken_classes.append(sorted_indices[0])

        return new_classes, faces

    def solve_conflicts_position(self, classes, faces, faces_positions):
        """
        Solve conflicts between classes (i.e. when we have two points classified to the same class)
        Using spatial position of the face in the image.

        Args:
            classes (np.ndarray): array of shape (n_samples)
                                  the original classes of the test faces.
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The original data to classify.
            faces_positions (np.ndarray): array of shape (n_samples, 2).
                                          The spatial position of the faces in the image.

        Returns:
            classes (np.ndarray): array of shape (n_samples)
                                  the new classes of the faces after solving the conflicts.
            faces (np.ndarray): array of shape (n_samples, n_features).
                                The new data after solving the conflicts.
        """

        new_classes = np.full_like(classes, fill_value=-1, dtype=np.int16)

        for i, row in enumerate(faces_positions):
            distances_to_class_i = np.linalg.norm(self.classes_spatial_positions - row, axis=1)
            sorted_indices = np.argsort(distances_to_class_i)

            if distances_to_class_i[sorted_indices[0]] >= 150:
                new_classes, label, faces = self.create_new_class(i, faces[i], new_classes, faces, faces_positions)
            else:
                new_classes[i] = sorted_indices[0]

        return new_classes, faces

    def create_new_class(self, i, row, classes, faces, faces_positions=None):
        """
        Create a new class for the current row (feature vector).

        Args:
            i (int): index of the current row in the data set.
            row (np.ndarray): feature vector of shape (n_features).
            classes (np.ndarray): labels array of shape (n_samples)
            faces (np.ndarray): array of shape (n_samples, n_features).
                                the faces' data set.
            faces_positions (np.ndarray): array of shape (n_samples, 2).
                                          The spatial position of each face in the image.

        Returns:
            classes (np.ndarray): updated labels array of shape (n_samples)
            label (int): the created label of the new class.
            faces (np.ndarray): updated data set of shape (n_samples, n_features).

        Notes:
            The new class will be created by not only adding the current row to it,
            instead, repeat the row `1 + k//2` times to make the class probable to be chosen.
        """

        label = self.n_classes
        classes[i] = label
        classes = np.append(classes, np.repeat(label, self.k // 2), axis=0)
        faces = np.append(faces, np.tile(row, (self.k // 2, 1)), axis=0)
        self.classes_count = np.append(self.classes_count, [1], axis=0)
        self.classes_centers = np.append(self.classes_centers, [row], axis=0)
        if faces_positions is not None:
            self.classes_spatial_positions = np.append(self.classes_spatial_positions, [faces_positions[i]], axis=0)
        self.n_classes += 1
        return classes, label, faces

    def incremental_update_class_info(self, label, index, faces, faces_positions, remove=False):
        """
        Update the class info, such as:
        - centers, which is the mean of all points belonging to the class.
        - spatial positions, which is the mean of all spatial positions in the original image belonging to the class.
        - count, which is the number of points belonging to the class.

        We do this incrementally, so we don't need to recalculate the class centers for each new point.

        Args:
            label (int): the class label.
            index (int): the index of the current row in the data set.
            faces (np.ndarray): array of shape (n_samples, n_features).
            faces_positions (np.ndarray): array of shape (n_samples, 2).
            remove (bool): if True, remove the point from the class.
                           else, add the point to the class, which is the default.
        """

        n = self.classes_count[label]
        op = np.subtract if remove else np.add

        self.classes_centers[label] = op(self.classes_centers[label] * n, faces[index]) / op(n, 1)
        self.classes_count[label] = op(self.classes_count[label], 1)

        if faces_positions is not None:
            original_position = self.classes_spatial_positions[label]
            self.classes_spatial_positions[label] = op(original_position * n, faces_positions[index]) / op(n, 1)

    def get_ids(self, faces: List[np.ndarray], faces_positions=None) -> npt.NDArray[np.uint16]:
        """
            This is the main function of the class.
            It takes a list of faces and returns the ids of the faces.

        Args:
            faces (List[np.ndarray]): list of grayscale face images of shape (n_samples, variable_image_size).
            faces_positions (np.ndarray): array of shape (n_samples, 4).
                                          The spatial position of each face in the image.
                                          It is used when conflict_solving_strategy is "position".
                                          If None, the conflict_solving_strategy will fall back to "min_distance".
        Returns:
            ids (np.array[int]): array of ids of shape (n_samples).
        """

        eigen_faces = eigen_faces_features(faces)

        if faces_positions is None:
            self.conflict_solving_strategy = "min_distance"
        else:
            faces_positions = self.get_midpoint(faces_positions)

        if self.classes is None:
            return self.knn_init(eigen_faces, faces_positions)
        return self.knn(eigen_faces, faces_positions)

    @staticmethod
    def get_midpoint(arr: np.ndarray):
        """
            Get the midpoint of points in the form of x1, y1, x2, y2.

        Args:
            arr (np.ndarray): array of shape (n_samples, 4).
                              Each row is a point in the form of x1, y1, x2, y2.
        Returns:
            midpoint (np.ndarray): array of shape (n_samples, 2).
        """

        if arr is None:
            return

        arr = arr.reshape(-1, 4)
        return np.array([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2]).T.reshape(-1, 2)

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

