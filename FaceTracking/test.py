import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

from utils import io
from utils import verbose
import face_track
import cv2


def test():
    args = io.get_command_line_args()
    input_video = args.input_video
    frames_to_process_each_second = args.fps

    video = io.read_video(input_video)

    video_fps = int(video.get(cv2.CAP_PROP_FPS))

    face_detector = cv2.CascadeClassifier(os.path.join(__CURRENT_DIR__, 'haarcascade_frontalface_default.xml'))

    frame_number = 0
    # Read frame by frame until video is completed
    while video.isOpened():
        frame_number += int(video_fps / frames_to_process_each_second)  # Process N every second

        ret, frame = video.read()
        if ret:  # Pipeline is implemented Here
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces: list = face_detector.detectMultiScale(frame_grey, scaleFactor=1.3, minNeighbors=5)

            # # Process each face, and draw a rectangle around it
            for (x, y, w, h) in faces:
                id = face_track.get_id(frame_grey, (x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person #{id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            verbose.imshow(frame, delay=1)
            verbose.print(f"Processing Frame #{frame_number}")

            # Seek the video to the required frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            verbose.print(f"[INFO] Video Current Time is {round(video.get(cv2.CAP_PROP_POS_MSEC), 3)} sec")

        else:
            break

    # When everything done, release the video capture object
    video.release()


if __name__ == '__main__':
    test()
