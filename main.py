import cv2
from utils import verbose
from utils import io
from utils import video_utils
from FaceDetection.HOG_SVM.main import FaceDetector

def main():
    args = io.get_command_line_args()
    input_video = args.input_video
    frames_to_process_each_second = args.fps

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

    ########################### Initialize FaceDetection ###########################
    modelPath = './FaceDetection/HOG_SVM/Models/ModelCBCL-HOG-TestSliding.sav'
    pcaPath = './FaceDetection/HOG_SVM/Models/PCAModelSliding.sav'
    faceDetector = FaceDetector(modelPath,pcaPath)

    frame_number = 0
    # Read frame by frame until video is completed
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        verbose.print(f"[INFO] Processing Frame #{frame_number}")

        # ============================================= Preprocessing =============================================

        # ============================================= Face Detection ============================================
        faces = faceDetector.detect(frame)
        verbose.print(f"[INFO] Detected {len(faces)} faces")
        for (x, y, lenX,lenY) in faces:
            cv2.rectangle(frame, (int(x), int(y)), (int(lenX), int(lenY)), (0, 255, 0), 2)
        # ========================================= Emotion Classification ========================================

        # ============================================= Face Tracking =============================================

        # ============================================ Gaze Estimation ============================================

        # ==================================== Profiling (Age/Gender Detection) ===================================

        # =========================================== Integrate Modules ===========================================

        # =============================================== Analytics ===============================================

        # =========================================================================================================
        
        verbose.imshow(frame, delay=1)

        frame_number += int(video_fps / frames_to_process_each_second)  # Process N every second

        # Seek the video to the required frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        verbose.print(f"[INFO] Video Current Time is {round(video.get(cv2.CAP_PROP_POS_MSEC), 3)} sec")

    # When everything done, release the video capture object
    video.release()


if __name__ == '__main__':
    main()

