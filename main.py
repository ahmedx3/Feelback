import cv2
from utils import verbose
from utils import io
from utils import video_utils
from FaceDetection.HOG_SVM.main import FaceDetector
from FaceTracking.knn import KNNIdentification
from AgeGenderClassification.main import GenderAgeClassification

import numpy as np

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
    modelPath = './FaceDetection/HOG_SVM/Models/Model_v3.sav'
    pcaPath = './FaceDetection/HOG_SVM/Models/PCA_v3.sav'
    faceDetector = FaceDetector(modelPath, pcaPath)

    ########################### Initialize FaceTracking ###########################
    faceTracker = KNNIdentification()

    ########################### Initialize Gender And Age ###########################
    modelAgePath = "./AgeGenderClassification/Models_Age/UTK_SVM_LPQ_47_44.model"
    modelGenderPath = "./AgeGenderClassification/Models_Gender/Kaggle_Tra_SVM_LPQ_87_86.model"
    genderPredictor = GenderAgeClassification(modelAgePath,modelGenderPath)

    # assign some unique colors for each face id for visualization purposes
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0), (128, 255, 0), (255, 128, 0)] * 10

    frame_number = 0
    # Read frame by frame until video is completed
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        verbose.print(f"[INFO] Processing Frame #{frame_number}")

        frame_number += int(video_fps / frames_to_process_each_second)  # Process N every second

        # Seek the video to the required frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # ============================================= Preprocessing =============================================
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ============================================= Face Detection ============================================
        faces = faceDetector.detect(frame)
        faces = np.array(faces, dtype=int)
        verbose.print(f"[INFO] Detected {faces.shape[0]} faces")

        if faces is None or len(faces) == 0:
            verbose.print("No Faces Detected, Skipping this frame")
            continue

        # ========================================= Emotion Classification ========================================

        # ============================================= Face Tracking =============================================
        ids = faceTracker.get_ids(frame_grey, faces)

        # ============================================ Gaze Estimation ============================================

        # ==================================== Profiling (Age/Gender Detection) ===================================
        genders = genderPredictor.getGender(frame_grey,faces)
        ages = genderPredictor.getAge(frame_grey,faces)

        # =========================================== Integrate Modules ===========================================

        # =============================================== Analytics ===============================================

        # ========================================== Verbosity Printing ===========================================

        if verbose.__VERBOSE__:
            # Draw a rectangle around each face with its person id
            for i in range(len(ids)):
                x1, y1, x2, y2 = faces[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=colors[ids[i]], thickness=2)
                cv2.putText(frame, f"Person #{ids[i]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ids[i]], 2)
                cv2.putText(frame, f"{genders[i]}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ids[i]], 2)
                cv2.putText(frame, f"{ages[i]} years", (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ids[i]], 2)

        verbose.imshow(frame, delay=1)

        verbose.print(f"[INFO] Video Current Time is {round(video.get(cv2.CAP_PROP_POS_MSEC), 3)} sec")

        # =========================================================================================================

    # When everything done, release the video capture object
    video.release()

    print(faceTracker.get_outliers_ids())


if __name__ == '__main__':
    main()

