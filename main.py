import cv2
from utils import verbose
from utils import io
from utils import video_utils


def main():
    args = io.get_command_line_args()
    input_video = args.input_video

    video = io.read_video(input_video)

    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = round(video_utils.get_duration(video), 3)

    verbose.print(f"[INFO] Video Resolution is {video_width}x{video_height}")
    verbose.print(f"[INFO] Video is running at {video_fps} fps")
    verbose.print(f"[INFO] Video has total of {video_frames_count} frames")
    verbose.print(f"[INFO] Video duration is {video_duration} sec")

    frame_number = 0
    # Read frame by frame until video is completed
    while video.isOpened():
        frame_number += 1

        ret, frame = video.read()
        if ret:  # Pipeline is implemented Here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            verbose.imshow(frame, delay=1)
            verbose.print(f"[INFO] Processing Frame #{frame_number}")

            verbose.print(f"[INFO] Video Current Time is {round(video.get(cv2.CAP_PROP_POS_MSEC), 3)} sec")

            # ============================================= Preprocessing =============================================

            # ============================================= Face Detection ============================================

            # ========================================= Emotion Classification ========================================

            # ============================================= Face Tracking =============================================

            # ============================================ Gaze Estimation ============================================

            # ==================================== Profiling (Age/Gender Detection) ===================================

            # =========================================== Integrate Modules ===========================================

            # =============================================== Analytics ===============================================


        else:
            break

    # When everything done, release the video capture object
    video.release()



if __name__ == '__main__':
    main()

