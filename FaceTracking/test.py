import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

from utils import io
from utils import verbose
import kmeans
import numpy as np
import cv2

__VERBOSE__ = verbose.__VERBOSE__


def test():
    args = io.get_command_line_args()
    input_video = args.input_video
    frames_to_process_each_second = args.fps

    video = io.read_video(input_video)

    video_fps = int(video.get(cv2.CAP_PROP_FPS))

    face_detector = cv2.CascadeClassifier(os.path.join(__CURRENT_DIR__, 'haarcascade_frontalface_default.xml'))
    face_track = None
    frame_number = 0
    # Read frame by frame until video is completed
    while video.isOpened():

        ok, frame = video.read()
        if not ok:
            break

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces: np.ndarray = face_detector.detectMultiScale(frame_grey, scaleFactor=1.3, minNeighbors=5)

        # Convert Bounding Boxes from (x, y, w, h) to (x1, y1, x2, y2)
        faces[:, 2] += faces[:, 0]
        faces[:, 3] += faces[:, 1]

        if face_track is None:
            face_track = kmeans.KmeansIdentification(k=faces.shape[0])

        ids = face_track.get_ids(frame_grey, faces)

        if __VERBOSE__:
            # Draw a rectangle around each face with its person id
            for i in range(len(ids)):
                x1, y1, x2, y2 = faces[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person #{ids[i]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        verbose.imshow(frame, delay=1)
        verbose.print(f"Processing Frame #{frame_number}")

        frame_number += int(video_fps / frames_to_process_each_second)  # Process N every second

        # Seek the video to the required frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        verbose.print(f"[INFO] Video Current Time is {round(video.get(cv2.CAP_PROP_POS_MSEC), 3)} sec")

    # When everything done, release the video capture object
    video.release()


if __name__ == '__main__':
    test()
