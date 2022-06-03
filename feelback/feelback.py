# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import cv2
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
import os

from .utils import verbose
from .AgeGenderClassification import AgeGenderClassification
from .FaceDetection import FaceDetector
from .FaceTracking import KNNIdentification
from .FacialExpressionRecognition import EmotionExtraction
from .GazeTracking import GazeEstimation
from .utils import io
from .utils import video_utils


class Feelback:

    __CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))

    # ==================================== Initialize FaceDetection ====================================
    modelPath = os.path.join(__CURRENT_DIR__, 'FaceDetection/HOG_SVM/Models/Model_v3.sav')
    pcaPath = os.path.join(__CURRENT_DIR__, 'FaceDetection/HOG_SVM/Models/PCA_v3.sav')
    faceDetector = FaceDetector(modelPath, pcaPath)

    # =================================== Initialize Emotion Extraction ==================================
    modelPath = os.path.join(__CURRENT_DIR__, "FacialExpressionRecognition/Models/Model.sav")
    emotionPredictor = EmotionExtraction(modelPath)

    def __init__(self, video_filename, fps, verbose_level=verbose.Level.INFO):
        verbose.set_verbose_level(verbose_level)

        self.frames_to_process_each_second = fps
        self.video = io.read_video(video_filename)
        video_fps = video_utils.get_fps(self.video, digits=3)
        self.frame_number_increment = round(video_fps / self.frames_to_process_each_second)

        width, height = video_utils.get_dimensions(self.video)
        verbose.info(f"Video Resolution is {width}x{height}")
        verbose.info(f"Video is running at {video_fps} fps")
        verbose.info(f"Video has total of {video_utils.get_number_of_frames(self.video)} frames")
        verbose.info(f"Video duration is {video_utils.get_duration(self.video, digits=3)} sec")

        # ==================================== Initialize FaceTracking ====================================
        self.faceTracker = KNNIdentification(conflict_solving_strategy="min_distance")

        # ================================== Initialize Gaze Estimation ===================================
        self.gazeEstimator = GazeEstimation()

        # =================================== Initialize Gender And Age =====================================
        modelAgePath = os.path.join(self.__CURRENT_DIR__, "AgeGenderClassification/Models_Age/UTK_SVR_LPQ_1030_1037.model")
        modelGenderPath = os.path.join(self.__CURRENT_DIR__, "AgeGenderClassification/Models_Gender/Kaggle_Tra_SVM_LPQ_86_84.model")
        self.genderPredictor = AgeGenderClassification(modelAgePath, modelGenderPath)

        self._persons = np.empty(0, dtype=[('person_id', int), ('age', int), ('gender', "U6")])
        self._data = np.empty(0, dtype=[('person_id', int), ('frame_number', int), ('face_position', int, 4),
                                        ('emotion', "U10"), ('attention', bool)])

        self.frame_number = 0

    @property
    def framerate(self):
        return self.frames_to_process_each_second

    @property
    def persons(self):
        return self._persons

    @property
    def emotions(self):
        return self._data[['person_id', 'frame_number', 'emotion']]

    @property
    def data(self):
        return self._data[['person_id', 'frame_number', 'emotion', 'attention']]

    @property
    def attention(self):
        return self._data[['person_id', 'frame_number', 'attention']]

    @property
    def video_frame_count(self):
        return video_utils.get_number_of_frames(self.video)

    @property
    def video_duration(self):
        return video_utils.get_duration(self.video)

    def progress(self):
        """
        Return the progress of the video
        """
        return min(100.0, round(100 * self.frame_number / self.video_frame_count, 2))

    def run(self):
        # Read frame by frame until video is completed
        while self.video.isOpened():
            try:
                ok, frame = self.video.read()
                if not ok:
                    break

                verbose.info(f"Processing Frame #{self.frame_number}")

                self.frame_number += self.frame_number_increment  # Process N every second

                # Seek the video to the required frame
                self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)

                # ========================================== Preprocessing ==========================================
                frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ========================================== Face Detection =========================================
                faces_positions = self.faceDetector.detect(frame)
                faces_positions = np.array(faces_positions, dtype=int)
                verbose.debug(f"Detected {faces_positions.shape[0]} faces")

                if faces_positions is None or len(faces_positions) == 0:
                    verbose.debug("No Faces Detected, Skipping this frame")
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
                self._append_data(ids, self.frame_number, faces_positions, emotions, gaze_attention)

                # ============================================ Analytics ============================================

                # ======================================= Verbosity Printing ========================================

                if verbose.is_verbose():
                    self._annotate_frame(frame, faces_positions, ids, ages, emotions, genders, gaze_attention)
                    verbose.imshow(frame, delay=1, level=verbose.Level.VISUAL)

                verbose.debug(f"Video Current Time is {round(video_utils.get_current_time(self.video), 3)} sec")

                # ===================================================================================================
            except KeyboardInterrupt:
                verbose.info("Ctrl-C Detected, Exiting")
                break
            except Exception as e:
                verbose.debug("Exception Occurred, Skipping this frame")
                verbose.error(e)
                verbose.print_exception_stack_trace()

        self.postprocessing()

    def _append_data(self, ids, frame_number, faces_positions, emotions, gaze_attention):
        new_data = np.empty(ids.shape[0], self._data.dtype)
        new_data['person_id'] = ids
        new_data['frame_number'] = frame_number
        new_data['face_position'] = faces_positions
        new_data['emotion'] = emotions
        new_data['attention'] = gaze_attention

        self._data = np.append(self._data, new_data)

    @staticmethod
    def _annotate_frame(frame, faces_positions, ids, ages, emotions, genders, gaze_attention):
        # assign some unique colors for each face id for visualization purposes
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0), (128, 255, 0), (255, 128, 0)]

        # Draw a rectangle around each face with its person id
        for i in range(len(ids)):
            color = colors[ids[i] % len(colors)]
            x1, y1, x2, y2 = faces_positions[i]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(frame, f"Person #{ids[i]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{genders[i]}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{ages[i]} years", (x1 + 150, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, emotions[i], (x1 + 150, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Attention: {gaze_attention[i]}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def postprocessing(self):
        """
        Remove Outlier persons, and gather data for analysis
        """

        outliers_ids = self.faceTracker.get_outliers_ids()
        valid_ids = self.faceTracker.get_valid_ids()

        genders = self.genderPredictor.getFinalGenders(valid_ids)
        ages = self.genderPredictor.getFinalAges(valid_ids)

        self._data = self._data[np.isin(self._data['person_id'], outliers_ids, invert=True)]
        self._persons = unstructured_to_structured(np.array([valid_ids, ages, genders]).T, self._persons.dtype)

    def save_postprocess_video(self, output_filename):
        if not output_filename:
            return

        output_video = video_utils.create_output_video(self.video, output_filename, self.framerate)
        frame_number = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Seek the video to the first frame
        while self.video.isOpened():
            ok, frame = self.video.read()
            if not ok:
                break

            frame_number += self.frame_number_increment
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Seek the video to the next frame

            frame_data = self._data[self._data['frame_number'] == frame_number]
            frame_persons_ids = frame_data['person_id']
            persons_data = self._persons[np.isin(self._persons['person_id'], frame_persons_ids)]
            self._annotate_frame(frame, frame_data['face_position'], frame_persons_ids, persons_data['age'],
                                 frame_data['emotion'], persons_data['gender'], frame_data['attention'])

            output_video.write(frame)

        output_video.release()

    def __del__(self):
        verbose.debug(f"Feelback Destructor is called, {self} will be deleted")
        self.video.release()
        # Closes all the frames
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = io.get_command_line_args()
    feelback = Feelback(args.input_video, args.fps, args.verbose)
    feelback.run()
    feelback.save_postprocess_video(args.output)

