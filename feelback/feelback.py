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
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
from skimage.feature import peak_local_max
import pickle


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

        self.video_filename = video_filename
        self.video = io.read_video(video_filename)
        self.video_fps = video_utils.get_fps(self.video, digits=3)
        self.frames_to_process_each_second = self.video_fps if fps == 'native' else int(fps)
        self.frame_number_increment = round(self.video_fps / self.frames_to_process_each_second)
        self.video_frame_count = video_utils.get_number_of_frames(self.video)
        self.video_duration = video_utils.get_duration(self.video, digits=3)

        width, height = video_utils.get_dimensions(self.video)
        verbose.info(f"Video Resolution is {width}x{height}")
        verbose.info(f"Video is running at {self.video_fps} fps")
        verbose.info(f"Video has total of {self.video_frame_count} frames")
        verbose.info(f"Video duration is {self.video_duration} sec")

        # ==================================== Initialize FaceTracking ====================================
        self.faceTracker = KNNIdentification(conflict_solving_strategy="min_distance")

        # ================================== Initialize Gaze Estimation ===================================
        self.gazeEstimator = GazeEstimation()

        # =================================== Initialize Gender And Age =====================================
        self.genderPredictor = AgeGenderClassification()

        self._persons = np.empty(0, dtype=[('person_id', int), ('age', int), ('gender', "U6")])
        self._frame_data = np.empty(0, dtype=[('frame_number', int), ('person_id', int), ('face_position', int, 4),
                                              ('emotion', "U10"), ('attention', bool)])
        self._time_data = np.empty(0, dtype=[('time', int), ('person_id', int), ('emotion', "U10"), ('attention', bool)])

        self._key_moments = np.empty((0, 2), dtype=int)
        self._key_moments_visualization_data = None

        # We have a minimum distance of 5 seconds between any two key moments
        # But for short videos, this distance is smaller according to video length
        self.key_moments_min_distance = int(max(1, min(5, self.video_duration / 5)) * self.frames_to_process_each_second)

        self.frame_number = 0

    @property
    def framerate(self):
        return self.frames_to_process_each_second

    @property
    def persons(self):
        return self._persons

    @property
    def key_moments_frames(self):
        return self._key_moments

    @property
    def key_moments_seconds(self):
        return self._key_moments / self.video_fps

    @property
    def emotions_frame_by_frame(self):
        return self._frame_data[['frame_number', 'person_id', 'emotion']]

    @property
    def emotions_second_by_second(self):
        return self._time_data[['time', 'person_id', 'emotion']]

    @property
    def data_frame_by_frame(self):
        return self._frame_data[['frame_number', 'person_id', 'emotion', 'attention', 'face_position']]

    @property
    def data_second_by_second(self):
        return self._time_data[['time', 'person_id', 'emotion', 'attention']]

    @property
    def attention_frame_by_frame(self):
        return self._frame_data[['frame_number', 'person_id', 'attention']]

    @property
    def attention_second_by_second(self):
        return self._time_data[['time', 'person_id', 'attention']]

    @property
    def faces_positions_frame_by_frame(self):
        return self._frame_data[['frame_number', 'person_id', 'face_position']]

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

                verbose.info(f"Processing Frame #{self.frame_number} ({self.progress()}%)")

                self.frame_number += self.frame_number_increment  # Process N frames every second

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
                frame_number = self.frame_number - self.frame_number_increment
                self._append_data(frame_number, ids, faces_positions, emotions, gaze_attention)

                # ============================================ Analytics ============================================

                # ======================================= Verbosity Printing ========================================

                if verbose.verbosity_level >= verbose.Level.VISUAL:
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

        # Closes all the frames
        cv2.destroyAllWindows()

        self.postprocessing()

    def _append_data(self, frame_number, ids, faces_positions, emotions, gaze_attention):
        new_data = np.empty(ids.shape[0], self._frame_data.dtype)
        new_data['frame_number'] = frame_number
        new_data['person_id'] = ids
        new_data['face_position'] = faces_positions
        new_data['emotion'] = emotions
        new_data['attention'] = gaze_attention

        self._frame_data = np.append(self._frame_data, new_data)

    @staticmethod
    def _annotate_frame(frame, faces_positions, ids, ages, emotions, genders, gaze_attention, annotations=None):
        if annotations is None:
            annotations = ['ids', 'age', 'gender', 'emotions', 'attention']

        # assign some unique colors for each face id for visualization purposes
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0), (128, 255, 0), (255, 128, 0)]

        # Draw a rectangle around each face with its person id
        for i in range(len(ids)):
            font, font_scale, color, line_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[ids[i] % len(colors)], 2
            text_format = font, font_scale, color, line_thickness

            x1, y1, x2, y2 = faces_positions[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)

            if 'ids' in annotations:
                cv2.putText(frame, f"Person #{ids[i]}", (x1, y1), *text_format)

            if 'gender' in annotations:
                cv2.putText(frame, f"{genders[i]}", (x1, y1 - 30), *text_format)

            if 'age' in annotations:
                cv2.putText(frame, f"{ages[i]} years", (x1 + 150, y1), *text_format)

            if 'emotions' in annotations:
                cv2.putText(frame, emotions[i], (x1 + 150, y1 - 30), *text_format)

            if 'attention' in annotations:
                cv2.putText(frame, f"Attention: {gaze_attention[i]}", (x1, y1 - 60), *text_format)

    def postprocessing(self):
        """
        Remove Outlier persons, generate key moments, and gather data for analysis
        """

        outliers_ids = self.faceTracker.get_outliers_ids()
        valid_ids = self.faceTracker.get_valid_ids()

        verbose.debug(f"Outlier ids: {outliers_ids}")

        genders = self.genderPredictor.getFinalGenders(valid_ids)
        ages = self.genderPredictor.getFinalAges(valid_ids)

        self._frame_data = self._frame_data[np.isin(self._frame_data['person_id'], outliers_ids, invert=True)]
        self._persons = unstructured_to_structured(np.array([valid_ids, ages, genders]).T, self._persons.dtype)
        self.normalize_frame_data()

        self.generate_key_moments()

    def normalize_frame_data(self):
        """
        Normalize frame data to prevent huge fluctuations, by grouping data by time in seconds
        For each second, calculate the most common emotion, and attention throughout all frames in that second
        """

        time = 0
        frame_number = 0
        for i in range(self._frame_data.shape[0]):
            end_frame_number = frame_number + int(self.video_fps)
            current_second_data = self._frame_data[(self._frame_data['frame_number'] >= frame_number) & (self._frame_data['frame_number'] < end_frame_number)]
            for person_id in self._persons['person_id']:
                person_data = current_second_data[current_second_data['person_id'] == person_id]

                if person_data.shape[0] == 0:
                    continue

                emotions, count = np.unique(person_data['emotion'], return_counts=True)

                # Emotion of the person is the most common emotion
                emotion = emotions[np.argmax(count)]

                # if person does pay attention in any frame, then attention is true for the whole second
                attention = np.any(person_data['attention'])

                time_data = np.array([(time, person_id, emotion, attention)], dtype=self._time_data.dtype)
                self._time_data = np.append(self._time_data, time_data)

            frame_number = end_frame_number
            time += 1

    def save_postprocess_video(self, output_filename, annotations=None):
        if not output_filename:
            return

        if self.video is None:
            verbose.error("No Video Loaded, Cannot Save Post Process Video")
            return

        verbose.info(f"Saving Output Video to '{output_filename}.mp4'")

        output_video = video_utils.create_output_video(self.video, output_filename, self.framerate)
        frame_number = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Seek the video to the first frame
        while self.video.isOpened():
            ok, frame = self.video.read()
            if not ok:
                break

            frame_data = self._frame_data[self._frame_data['frame_number'] == frame_number]
            frame_persons_ids = frame_data['person_id']

            # the following code is to make sure that the persons_data is in the same order as the frame_persons_ids
            persons_data_ids = np.ravel([np.where(np.isin(self._persons['person_id'], id)) for id in frame_persons_ids])

            persons_data = self._persons[persons_data_ids.astype(int)]
            self._annotate_frame(frame, frame_data['face_position'], frame_persons_ids, persons_data['age'],
                                 frame_data['emotion'], persons_data['gender'], frame_data['attention'], annotations)

            frame_number += self.frame_number_increment
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Seek the video to the next frame

            output_video.write(frame)

        output_video.release()

    @staticmethod
    def smooth_curve(data, window_size=30):
        """
        Smooth the data using a moving average window of size window_size.
        """

        if window_size >= data.size:
            window_size = int(data.size // 2 - 1)

        window_size = max(1, window_size)

        cumulative_sum = np.cumsum(data)
        smoothed = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        return np.concatenate([np.zeros(window_size // 2), smoothed, np.zeros(window_size // 2)])

    def get_key_moments_interval(self, histogram, local_max, local_min):
        """
        Get the start, end frames of each key moment.

        Args:
            histogram (np.ndarray): The histogram of the emotions in the video.
            local_max (np.ndarray): The local maxima of the histogram.
            local_min (np.ndarray): The local minima of the histogram.

        Returns:
            list[tuple(start, end)]: The start and end frames of each key moment.
        """

        key_moments_interval = np.empty((local_max.shape[0], 2), dtype=int)
        for i, key_moment in enumerate(local_max):
            local_min_before = local_min[np.searchsorted(local_min, key_moment, side='left') - 1]
            local_min_after = local_min[np.searchsorted(local_min, key_moment, side='right')]
            max_local_min = max(histogram[local_min_before], histogram[local_min_after])
            key_moment_value = histogram[key_moment]
            effective_local_min = key_moment_value - (key_moment_value - max_local_min)/2
            duration = np.argwhere(histogram[local_min_before:local_min_after] >= effective_local_min).ravel()
            key_moment_duration = local_min_before + duration
            start, end = key_moment_duration[0], key_moment_duration[-1]
            key_moments_interval[i] = start, end

        return key_moments_interval * self.frame_number_increment

    def generate_key_moments(self):
        if self._key_moments_visualization_data is not None:
            return self._key_moments_visualization_data

        emotions = {
            'surprise': 5,
            'happy': 3,
            'neutral': -1,
            'sadness': -3,
            'disgust': -5
        }

        emotions_arr = np.empty_like(self._frame_data, dtype=int)
        for emotion, value in emotions.items():
            emotions_arr[self._frame_data['emotion'] == emotion] = value

        histogram = np.zeros(int(np.ceil(self.video_frame_count / self.frame_number_increment)), dtype=int)

        for i in range(self._frame_data.shape[0]):
            histogram[self._frame_data['frame_number'][i] // self.frame_number_increment] += emotions_arr[i]

        histogram[0] = histogram[-1] = 0

        # Smooth the histogram
        histogram = self.smooth_curve(histogram, histogram.size // 10)
        histogram = self.smooth_curve(histogram, histogram.size // 10)

        distance = max(1, self.key_moments_min_distance - 1)
        local_max = peak_local_max(histogram, distance, threshold_abs=histogram.max() // 2, exclude_border=True).ravel()
        local_min = np.array([0, *peak_local_max(-histogram, max(1, distance // 5)).ravel(), histogram.size - 1], dtype=int)

        local_max.sort()
        local_min.sort()

        self._key_moments = self.get_key_moments_interval(histogram, local_max, local_min)

        verbose.debug("Key Moments:", [f"(start:{s:.3f}, end:{e:.3f})" for s, e in self.key_moments_seconds])

        self._key_moments_visualization_data = histogram, local_max, local_min
        return self._key_moments_visualization_data

    def visualize_key_moments(self, output_filename=None):
        if verbose.verbosity_level < verbose.Level.VISUAL and output_filename is None:
            return

        histogram, local_max, local_min = self.generate_key_moments()

        if len(self.key_moments_frames) == 0:
            return

        convert_to_seconds = self.frame_number_increment / self.video_fps

        plt.figure(dpi=500)

        try:
            # Sets the number of ticks on the x-axis.
            plt.gca().xaxis.set_major_locator(plt_ticker.MultipleLocator(int(self.video_duration / 10)))
            plt.gca().xaxis.set_minor_locator(plt_ticker.MultipleLocator(int(self.video_duration / 20)))
        except:
            pass

        margin_label = 'Key Moment Margin'
        range_label = 'Key Moment Range'
        for start, end in self.key_moments_seconds:
            s = int(start / convert_to_seconds)
            e = int(end / convert_to_seconds)
            plt.vlines(x=start, ymin=0, ymax=histogram[s], color='r', linewidth=2, alpha=0.5, linestyles='dashed', label=margin_label)
            plt.fill_between(convert_to_seconds * np.arange(s, e), histogram[s:e], alpha=0.5, color='y', label=range_label)
            plt.vlines(x=end, ymin=0, ymax=histogram[e], color='r', linewidth=2, alpha=0.5, linestyles='dashed')
            margin_label = range_label = None

        plt.grid(visible=True, which='both', color="grey", linewidth="1", alpha=0.2, linestyle="-.")
        plt.hlines(0, 0, self.video_duration, color='black', linewidth=1)

        plt.plot(convert_to_seconds * np.arange(histogram.size), histogram, c='b')
        plt.scatter(convert_to_seconds * local_max, histogram[local_max], c='g', marker='x', zorder=10, label='Local Maxima')
        plt.scatter(convert_to_seconds * local_min, histogram[local_min], c='r', marker='o', label='Local Minima')
        plt.legend()
        plt.xlabel("Time in Seconds")
        plt.ylabel("Total Emotion Value")
        plt.title("Key Moments Visualization")

        if output_filename is not None:
            verbose.info(f"Saving Key Moments Visualization to '{output_filename}'")
            plt.savefig(output_filename)
        plt.show() if verbose.verbosity_level >= verbose.Level.VISUAL else None

    def __del__(self):
        verbose.debug(f"Feelback Destructor is called, {self} will be deleted")
        if self.video is not None:
            self.video.release()
            self.video = None

    def __getstate__(self):
        """
        Returns a dictionary representing the state of the object.
        Used for serialization. (Pickle.dump)
        """

        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['video']
        return state

    def __setstate__(self, state):
        """
        Sets the state of the object.
        Used for deserialization. (Pickle.load)
        """

        self.__dict__.update(state)

        try:
            self.video = io.read_video(self.video_filename)
        except Exception as e:
            self.video = None
            print(f"[WARNING] Can not restore video file: {e}")


def main():
    args = io.get_command_line_args()
    feelback = Feelback(args.input_video, args.fps, args.verbose)
    feelback.run()
    feelback.save_postprocess_video(args.output, args.output_annotations)
    feelback.visualize_key_moments(args.output_key_moments)

    if args.dump is not None:
        verbose.info(f"Dumping Feelback object to '{args.dump}'")
        with open(args.dump, 'wb') as f:
            pickle.dump(feelback, f)


"""
Pickle SHOULD NEVER BE USED TO DUMP/LOAD DATA FROM '__main__' MODULE
BECAUSE IT NOT BE PORTABLE AND WILL NOT BE LOADED FROM DIFFERENT MODULES

Therefore calling this script as __main__ is meant to do nothing.
To use feelback as a module, call feelback.main() instead, same as `feelback_cli.py` does
"""
if __name__ == '__main__':
    print("Feelback is a module, please call `feelback.main()` instead")
    print("Use `feelback_cli.py` to run Feelback CLI, or use our web interface")
    exit(1)

