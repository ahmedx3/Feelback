"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

from utils import io
from utils import verbose
import numpy as np
import cv2
import dlib
from gaze_tracking import GazeTracking

__VERBOSE__ = verbose.__VERBOSE__


def test():
    args = io.get_command_line_args()
    input_video = args.input_video
    frames_to_process_each_second = args.fps

    video = io.read_video(input_video)

    video_fps = int(video.get(cv2.CAP_PROP_FPS))

    face_detector = dlib.get_frontal_face_detector()
    gaze = GazeTracking()

    frame_number = 0
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_rectangles = face_detector(frame_grey)
        faces = np.array([[face.left(), face.top(), face.right(), face.bottom()] for face in faces_rectangles])

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame_grey, faces[0])

        if __VERBOSE__:
            # Draw a rectangle around each face with its person id

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()

            cv2.circle(frame, left_pupil, radius=5, color=(0, 255, 0), thickness=1)
            cv2.circle(frame, right_pupil, radius=5, color=(0, 255, 0), thickness=1)

            cv2.putText(frame, f"Left pupil: {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            cv2.putText(frame, gaze.get_current_state_text(), (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        verbose.imshow(frame, delay=1)
        verbose.print(f"Processing Frame #{frame_number}")

        frame_number += int(video_fps / frames_to_process_each_second)  # Process N every second

        # Seek the video to the required frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        verbose.print(f"[INFO] Video Current Time is {round(video.get(cv2.CAP_PROP_POS_MSEC), 3)} sec")

    # When everything done, release the video capture object
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
