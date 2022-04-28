import os
import sys

# Allow importing modules from parent directory
# TODO: Use a more clean approach as modules
__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)
sys.path.append(__PARENT_DIR__)

import knn
import cv2
import dlib
import numpy as np
import itertools
from datetime import datetime
import argparse


face_detector = dlib.get_frontal_face_detector()
test_dir = ""
faces_db = {}


def test(number_of_faces):
    count = 0
    count_errors = 0

    dir_combinations = itertools.combinations(sorted(os.listdir(test_dir)), r=number_of_faces)
    for dirs in dir_combinations:
        local_count = 0
        local_count_errors = 0

        print(*dirs, sep=" vs ")

        all_files = []
        for d in dirs:
            files = sorted(os.listdir(os.path.join(test_dir, d)))
            all_files += [[os.path.join(d, f) for f in files]]

        face_track = knn.KNNIdentification()

        for files in zip(*all_files):

            faces = [get_face(f) for f in files]

            ids = face_track.get_ids(faces)
            local_count += 1
            if not np.all(ids == range(number_of_faces)):
                local_count_errors += 1

        count += local_count
        count_errors += local_count_errors

        print(f"Local Accuracy: {100 - 100 * (local_count_errors / local_count)}%")
        print(f"Total Accuracy: {100 - 100 * (count_errors / count)}%")


def get_face(file):
    face = faces_db.get(file, None)
    if face is None:
        face = load_face(file)
        faces_db[file] = face

    return face


def load_face(file):
    frame_grey = cv2.imread(os.path.join(test_dir, file), cv2.IMREAD_GRAYSCALE)
    face = face_detector(frame_grey)[0]
    face = frame_grey[face.top():face.bottom(), face.left():face.right()]
    return face


def get_command_line_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('test_dir', help="Test Set Directory")
    args_parser.add_argument("-n", "--number-of-faces", help="Number of faces", default=2, type=int, metavar='N')
    return args_parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    args = get_command_line_args()
    test_dir = args.test_dir
    test(args.number_of_faces)
    end = datetime.now()
    print(f"Total Time: {end - start}")
