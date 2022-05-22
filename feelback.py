# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import cv2
import numpy as np

from .utils import verbose
from .AgeGenderClassification import AgeGenderClassification
from .FaceDetection import FaceDetector
from .FaceTracking import KNNIdentification
from .FacialExpressionRecognition import EmotionExtraction
from .GazeTracking import GazeEstimation
from .utils import io
from .utils import video_utils


class Feelback:

    # ==================================== Initialize FaceDetection ====================================
    modelPath = './FaceDetection/HOG_SVM/Models/Model_v3.sav'
    pcaPath = './FaceDetection/HOG_SVM/Models/PCA_v3.sav'
    faceDetector = FaceDetector(modelPath, pcaPath)

    # =================================== Initialize Emotion Extraction ==================================
    modelPath = "./FacialExpressionRecognition/Models/Model.sav"
    emotionPredictor = EmotionExtraction(modelPath)

    # =================================== Initialize Gender And Age =====================================
    modelAgePath = "./AgeGenderClassification/Models_Age/UTK_SVR_LPQ_1030_1037.model"
    modelGenderPath = "./AgeGenderClassification/Models_Gender/Kaggle_Tra_SVM_LPQ_86_84.model"
    genderPredictor = AgeGenderClassification(modelAgePath, modelGenderPath)

    def __init__(self, video_filename, fps, verbosity=False):
        verbose.__VERBOSE__ = verbosity

        self.frames_to_process_each_second = fps
        self.video = io.read_video(video_filename)
        self.video_fps = int(self.video.get(cv2.CAP_PROP_FPS))

        video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frames_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = round(video_utils.get_duration(self.video), 3)

        verbose.print(f"[INFO] Video Resolution is {video_width}x{video_height}")
        verbose.print(f"[INFO] Video is running at {self.video_fps} fps")
        verbose.print(f"[INFO] Video has total of {video_frames_count} frames")
        verbose.print(f"[INFO] Video duration is {video_duration} sec")

        # ==================================== Initialize FaceTracking ====================================
        self.faceTracker = KNNIdentification(conflict_solving_strategy="min_distance", verbosity=verbosity)

        # ================================== Initialize Gaze Estimation ===================================
        self.gazeEstimator = GazeEstimation()

    def run(self):
        frame_number = 0
        # Read frame by frame until video is completed
        while self.video.isOpened():
            try:
                ok, frame = self.video.read()
                if not ok:
                    break

                verbose.print(f"[INFO] Processing Frame #{frame_number}")

                frame_number += round(self.video_fps / self.frames_to_process_each_second)  # Process N every second

                # Seek the video to the required frame
                self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # ========================================== Preprocessing ==========================================
                frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ========================================== Face Detection =========================================
                faces_positions = self.faceDetector.detect(frame)
                faces_positions = np.array(faces_positions, dtype=int)
                verbose.print(f"[INFO] Detected {faces_positions.shape[0]} faces")

                if faces_positions is None or len(faces_positions) == 0:
                    verbose.print("[WARNING] No Faces Detected, Skipping this frame")
                    continue

                faces = []
                for x1, y1, x2, y2 in faces_positions:
                    faces.append(frame_grey[y1:y2, x1:x2])

                # ====================================== Emotion Classification =====================================
                emotions = self.emotionPredictor.getEmotion(faces)

                # ========================================== Face Tracking ==========================================
                ids = self.faceTracker.get_ids(faces)

                # ========================================= Gaze Estimation =========================================
                gaze_attention = self.gazeEstimator.get_gaze_attention(frame_grey, faces_positions, ids)

                # ================================= Profiling (Age/Gender Detection) ================================
                genders = self.genderPredictor.getGender(frame_grey, faces_positions, ids)
                ages = self.genderPredictor.getAge(frame_grey, faces_positions, ids)

                # ======================================== Integrate Modules ========================================

                # ============================================ Analytics ============================================

                # ======================================= Verbosity Printing ========================================

                if verbose.__VERBOSE__:
                    self.__imshow(frame, faces_positions, ids, ages, emotions, genders, gaze_attention)

                verbose.print(f"[INFO] Video Current Time is {round(video_utils.get_current_time(self.video), 3)} sec")

                # ===================================================================================================
            except KeyboardInterrupt:
                print('KeyboardInterrupt exception. EXITING')
                break
            except:
                verbose.print("[ERROR] Exception Occurred, Skipping this frame")

        # When everything done, release the video capture object
        self.video.release()
        cv2.destroyAllWindows()

        print(self.faceTracker.get_outliers_ids())

    @staticmethod
    def __imshow(frame, faces_positions, ids, ages, emotions, genders, gaze_attention):
        # assign some unique colors for each face id for visualization purposes
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0), (128, 255, 0), (255, 128, 0)] * 10

        # Draw a rectangle around each face with its person id
        for i in range(len(ids)):
            color = colors[ids[i]]
            x1, y1, x2, y2 = faces_positions[i]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(frame, f"Person #{ids[i]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{genders[i]}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{ages[i]} years", (x1 + 150, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, emotions[i], (x1 + 150, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Attention: {gaze_attention[i]}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        verbose.imshow(frame, delay=1)


if __name__ == '__main__':
    args = io.get_command_line_args()
    feelback = Feelback(args.input_video, args.fps, args.verbose)
    feelback.run()

