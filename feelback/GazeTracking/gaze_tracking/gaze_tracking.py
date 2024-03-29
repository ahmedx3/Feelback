from __future__ import division
import os
import dlib
import numpy as np
from .eye import Eye
from .calibration import Calibration

__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    # _predictor is used to get facial landmarks of a given face
    # Use as static class variable to share the model between all objects to save memory and time during initialization
    _predictor = dlib.shape_predictor(os.path.join(__CURRENT_DIR__, "../../Shared_models/shape_predictor_68_face_landmarks.dat"))

    def __init__(self, right_threshold: float = 0.35, left_threshold: float = 0.75, blinking_threshold: float = 3.8):
        """
        Create Gaze Tracking Object

        Initializes Hyper-Parameters for considering looking left, right, and center.
        Horizontal looking ratio is a number between 0.0 and 1.0 that indicates the direction of the gaze.
        The extreme right is 0.0, the center is 0.5 and the extreme left is 1.0

        Values less than right_threshold are considered right, values greater than left_threshold are considered left,
        Values in-between are considered center.

        Arguments:
            right_threshold (float): Hyper-Parameter to consider person is looking towards right.
            left_threshold (float):  Hyper-Parameter to consider person is looking towards left.
            blinking_threshold (float):  Hyper-Parameter to consider person is blinking.
        """

        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.blinking_threshold = blinking_threshold
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self, face):
        """Initialize Eye objects

        Arguments:
            face (tuple): Face bounding box (x1, y1, x2, y2)
        """

        try:
            landmarks = self._predictor(self.frame, dlib.rectangle(*face))
            self.eye_left = Eye(self.frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(self.frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame, face):
        """Refreshes the frame and analyzes it.

        Arguments:
            face (tuple): Face bounding box (x1, y1, x2, y2)
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze(face)

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return x, y

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return x, y

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= self.right_threshold

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= self.left_threshold

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > self.blinking_threshold

    def get_current_state_text(self):
        """
        Returns the text description of the eyes current state
        """

        text = ""
        if self.is_blinking():
            text = "Blinking"
        elif self.is_right():
            text = "Looking right"
        elif self.is_left():
            text = "Looking left"
        elif self.is_center():
            text = "Looking center"
        return text


class GazeEstimation:
    """
    Wrapper Class around GazeTracking to support multiple faces
    """

    def __init__(self, right_threshold: float = 0.35, left_threshold: float = 0.75, blinking_threshold: float = 3.8):
        """
        Create Gaze Tracking Object

        Initializes Hyper-Parameters for considering looking left, right, and center.
        Horizontal looking ratio is a number between 0.0 and 1.0 that indicates the direction of the gaze.
        The extreme right is 0.0, the center is 0.5 and the extreme left is 1.0

        Values less than right_threshold are considered right, values greater than left_threshold are considered left,
        Values in-between are considered center.

        Notes:
            This class uses a separate GazeTracking object for each user, because each user may have different type of
            eyes, where combining all of them in single object can mess the calibration.

        Arguments:
            right_threshold (float): Hyper-Parameter to consider person is looking towards right.
            left_threshold (float):  Hyper-Parameter to consider person is looking towards left.
            blinking_threshold (float): Hyper-Parameter to consider person is blinking.
        """

        self.right_threshold = right_threshold
        self.left_threshold = left_threshold
        self.blinking_threshold = blinking_threshold
        self.gazes = []

    def get_gaze_attention(self, frame: np.ndarray, faces_positions: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """
        Check for each user whether he is paying attention or not, using where he is looking.

        Arguments:
            frame (np.ndarray): The frame to process.
            faces_positions (np.ndarray): Array of face bounding boxs (x1, y1, x2, y2)
            ids (np.ndarray): Array of ids of the corresponding faces (used to identify the eyes of each user)

        Returns: Array of booleans which indicates for each user whether he is paying attention or not.

        """

        gaze_attention = np.zeros(faces_positions.shape[0], dtype=bool)
        while ids.max(initial=0) >= len(self.gazes):
            self.gazes.append(GazeTracking(self.right_threshold, self.left_threshold, self.blinking_threshold))

        for i in range(ids.shape[0]):
            self.gazes[ids[i]].refresh(frame, faces_positions[i])
            gaze_attention[i] = self.gazes[ids[i]].is_center()

        return gaze_attention

    def get_gaze_state_text(self, ids: np.ndarray):
        gaze_text_descriptions = []

        for id in ids:
            gaze_text_descriptions.append(self.gazes[id].get_current_state_text())
        return gaze_text_descriptions
