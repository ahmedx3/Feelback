import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

from utils.img_utils import BoundingBox
from utils import verbose

from FaceTracking.feature_extracton import eigen_faces_features
import numpy as np
import numpy.typing as npt


class KNNIdentification:
    """
    This class use a modified version of K Nearest Neighbours (KNN) Classification Algorithm to use in online
    face recognition and identification.
    This class does not need labels, instead it labels the data automatically.

    Notes:
        At the very beginning, the number of available data may be smaller than k, therefore all points are considered
        we will probably have a tie.
        Instead, we choose, a number smaller than k according the current amount of data, given by this formula
        `k = minimum(k, 1 + n_data//k)`
    """

    def __init__(self, k=5):
        """

        Args:
            k (int): Number of neighbours to use when classifying. Defaults to 5.
        """

        self.k = k
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

        self.features = x
        self.classes = np.arange(x.shape[0])

        return self.classes

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
            At the very beginning, the number of available data may be smaller than k, therefore all points are considered
            we will probably have a tie.
            Instead, we choose, a number smaller than k according the current amount of data, given by this formula
            `k = minimum(k, 1 + n_data//k)`
        """

        classes = np.zeros(x.shape[0], dtype=np.uint16)

        for i, row in enumerate(x):
            distances = np.linalg.norm(self.features - row, axis=1)
            k = min(self.k, 1 + int(self.features.shape[0] // self.k))
            nearest_k_classes = self.classes[np.argsort(distances)][:k]
            label = np.bincount(nearest_k_classes).argmax()  # get the most frequent class
            classes[i] = label

        self.features = np.append(self.features, x, axis=0)
        self.classes = np.append(self.classes, classes, axis=0)

        return classes

    def get_ids(self, frame: np.ndarray, faces_positions: npt.NDArray[BoundingBox]) -> npt.NDArray[np.uint16]:
        # TODO: Feature Extraction from faces (Eigen-faces + position)
        eigen_faces = eigen_faces_features(frame, faces_positions)
        return self.knn_init(eigen_faces) if self.classes is None else self.knn(eigen_faces)
