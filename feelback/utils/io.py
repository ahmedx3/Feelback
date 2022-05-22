import argparse
import os
import cv2
import numpy as np


def get_command_line_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('input_video', help="Input Video File Path")
    args_parser.add_argument("-f", "--fps", help="Process N frames every second", default=3, type=int, metavar='N')
    args_parser.add_argument("-v", "--verbose", help="Enable more verbosity", default=False, action="store_true")
    return args_parser.parse_args()


def get_filenames(path: str) -> list:
    if os.path.isfile(path):
        return [os.path.basename(path)]

    if os.path.isdir(path):
        return os.listdir(path)


def write_image(img: np.ndarray, directory: str, filename: str):
    if os.path.isdir(directory) and filename is not None:
        filename = os.path.join(directory, filename)
        cv2.imwrite(filename, img)

    elif filename is not None:
        os.mkdir(directory)
        filename = os.path.join(directory, filename)
        cv2.imwrite(filename, img)


def read_video(input_path: str) -> cv2.VideoCapture:
    video = cv2.VideoCapture(input_path)

    if not video.isOpened(): 
        print(f"Error opening video file at '{input_path}'")
        exit(-1)
    
    return video


def read_grayscale_image(input_path: str, filename: str) -> np.ndarray:
    if os.path.isfile(input_path):
        return cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    elif os.path.isdir(input_path) and filename is not None:
        filename = os.path.join(input_path, filename)
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

