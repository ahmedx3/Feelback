# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

from ..utils import io, verbose, video_utils
import knn
import numpy as np
import cv2
import dlib


def test():
    args = io.get_command_line_args()
    input_video = args.input_video
    frames_to_process_each_second = args.fps
    __VERBOSE__ = verbose.__VERBOSE__ = args.verbose

    video = io.read_video(input_video)

    video_fps = int(video.get(cv2.CAP_PROP_FPS))

    face_detector = dlib.get_frontal_face_detector()
    face_track = knn.KNNIdentification()
    frame_number = 0
    # assign some unique colors for each face id for visualization purposes
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0), (128, 255, 0), (255, 128, 0)] * 10

    # Read frame by frame until video is completed
    while video.isOpened():

        ok, frame = video.read()
        if not ok:
            break

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_rectangles = face_detector(frame_grey, 0)
        faces_positions = [[face.left(), face.top(), face.right(), face.bottom()] for face in faces_rectangles]
        faces = [frame_grey[y1:y2, x1:x2] for x1, y1, x2, y2 in faces_positions]

        if faces is not None and len(faces):

            ids = face_track.get_ids(faces)

            if __VERBOSE__:
                # Draw a rectangle around each face with its person id
                for i in range(len(ids)):
                    x1, y1, x2, y2 = faces_positions[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[ids[i]], 2)
                    cv2.putText(frame, f"Person #{ids[i]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ids[i]], 2)

        verbose.imshow(frame, delay=1)
        verbose.print(f"Processing Frame #{frame_number}")

        frame_number += round(video_fps / frames_to_process_each_second)  # Process N every second

        # Seek the video to the required frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        verbose.print(f"[INFO] Video Current Time is {round(video_utils.get_current_time(video), 3)} sec")

    # When everything done, release the video capture object
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
