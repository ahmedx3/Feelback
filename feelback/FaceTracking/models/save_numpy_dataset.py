#!/usr/bin/env python3

import numpy as np
import cv2
import os
import dlib
from datetime import datetime
import argparse
import traceback


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--numpy_output', type=str, required=True)
    parser.add_argument('--double', help="Make two face detections after each other", default=False, action="store_true")
    parser.add_argument('--size', help="Number of samples to taken", type=int, default=45)
    return parser.parse_args()


args = get_args()
face_detector = dlib.get_frontal_face_detector()
np.random.seed(42)


def get_area(rectangle):
    if rectangle is None:
        return 0
    if type(rectangle) is dlib.rectangle:
        return (rectangle.right() - rectangle.left()) * (rectangle.bottom() - rectangle.top())
    if type(rectangle) is np.ndarray:
        return rectangle.shape[0] * rectangle.shape[1]


def get_max_face(faces):
    max_face = None
    max_area = -1
    for face in faces:
        area = get_area(face)
        if area > max_area:
            max_area = area
            max_face = face
    return max_face


def get_area_ratio(face1, face2):
    area1 = get_area(face1)
    area2 = get_area(face2)
    if area1 == 0 or area2 == 0:
        return 0
    max_area = max(area1, area2)
    min_area = min(area1, area2)
    return min_area / max_area


def reload_face(face):
    frame_grey = np.copy(face)
    face = get_max_face(face_detector(frame_grey, 3))
    if face is None:
        return frame_grey
    face = frame_grey[face.top():face.bottom(), face.left():face.right()]
    return face


def load_face(file):
    frame_grey = cv2.imread(file)
    face = get_max_face(face_detector(frame_grey, 3))
    face = frame_grey[face.top():face.bottom(), face.left():face.right()]

    new_face = reload_face(face) if args.double else None
    if get_area_ratio(face, new_face) > 0.5:
        face = new_face

    return face


def load_dataset(path_to_dataset, output_path, numpy_output_path):
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)

    files = list(os.listdir(path_to_dataset))
    print(f'Dataset Size: {len(files)}')

    for file in files:
        path = os.path.join(path_to_dataset, file)
        try:
            face = load_face(path)
            cv2.imwrite(os.path.join(output_path, file), face)

            face = cv2.resize(face, (48, 48)).ravel()
            np.save(os.path.join(numpy_output_path, file), face)

        except Exception as e:
            print(f"ERROR at '{path}'", e)
            traceback.print_exc()

    print('Finished saving dataset.')


def filter_numpy_dataset(numpy_output_path):
    files = list(os.listdir(numpy_output_path))
    random_files = np.random.choice(files, size=args.size, replace=False)
    for file in files:
        if file not in random_files:
            os.remove(os.path.join(numpy_output_path, file))


start = datetime.now()
print(f"Loading dataset {args.dataset}")
load_dataset(args.dataset, args.output, args.numpy_output)
filter_numpy_dataset(args.numpy_output)
print(f"Time Taken: {datetime.now() - start}")
