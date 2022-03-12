import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

from utils import verbose
from utils import img_utils
from utils.img_utils import BoundingBox
import numpy as np
import numpy.typing as npt
import cv2

dtype = np.dtype([('tracker', cv2.TrackerCSRT), ('last-seen-frame', int), ('last-seen-position', int, 4)])

database = np.empty(shape=0, dtype=dtype)
IOU_THRESHOLD = 0.5


def init(frame: np.ndarray, faces_positions: npt.NDArray[BoundingBox]):
    for face_position in faces_positions:
        add_new_face(frame, face_position, 0)


def get_ids(frame: np.ndarray, faces_positions: npt.NDArray[BoundingBox], frame_number: int) -> npt.NDArray[int]:
    global database

    ids = np.empty(faces_positions.shape[0], dtype=int)

    trackers: np.ndarray = database['tracker']
    for i in range(trackers.shape[0]):
        success, bounding_box = trackers[i].update(frame)
        if success:
            database[i]['last-seen-position'] = img_utils.to_x1_y1_x2_y2_bounding_box(bounding_box)

    for i, face_position in enumerate(faces_positions):
        iou = img_utils.intersection_over_union(face_position, database['last-seen-position'])
        id = np.argmax(iou)
        if iou[id] > IOU_THRESHOLD:
            ids[i] = id
            database[id]['last-seen-position'] = face_position
            database[id]['last-seen-frame'] = frame_number
        else:
            #  Create new face
            ids[i] = len(database)
            add_new_face(frame, face_position, frame_number)

    return ids


def add_new_face(frame: np.ndarray, face_position: BoundingBox, frame_number: int):
    global database

    tracker: cv2.TrackerCSRT = cv2.TrackerCSRT_create()
    tracker.init(frame, img_utils.to_x_y_w_h_bounding_box(face_position))

    face_data = np.array([(tracker, frame_number, face_position)], dtype=dtype)
    database = np.append(database, face_data)
