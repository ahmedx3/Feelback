import argparse
import os
import cv2
import numpy as np

_annotations = ['all', 'none', 'ids', 'age', 'gender', 'emotions', 'attention']


def _fps_check(fps):
    if fps == 'native':
        return fps
    else:
        try:
            return int(fps)
        except ValueError:
            raise argparse.ArgumentTypeError(f"FPS must be an integer or 'native'")


def get_command_line_args():
    args_parser = argparse.ArgumentParser(description="Feelback is An Automated Video Feedback Framework",
                                          formatter_class=argparse.RawTextHelpFormatter)

    args_parser.add_argument('input_video', help="Input Video File Path")
    args_parser.add_argument("-o", "--output", help="Save processed video to filename.mp4", metavar='filename')

    args_parser.add_argument("--output-annotations", nargs="+", choices=_annotations, metavar='annotations',
                             help=f"Which annotations to add to processed video\n" +
                                  f"annotations can be any of: {_annotations}\n" +
                                  f"You can add multiple annotations separated by space\n" +
                                  f"Default: all\n\n", default=['all'])

    g = args_parser.add_mutually_exclusive_group()
    g.add_argument("--dump", help="Dump Feelback Object After Processing", type=str, default=None, metavar='filename')
    g.add_argument("--load", help="Load Dumped Feelback Object [For Debug]", type=str, default=None, metavar='filename')

    args_parser.add_argument("--output-key-moments", help="Save Key Moments Visualizations to file", metavar='filename')
    args_parser.add_argument("-f", "--fps", help="Process N frames every second, Or `native` to process all frames",
                             default=3, type=_fps_check, metavar='N | native')
    args_parser.add_argument("-v", "--verbose", help="Enable more verbosity", default=0, action="count")
    args = args_parser.parse_args()
    args.output_annotations = process_annotation_flags(args.output_annotations)
    return args


def process_annotation_flags(annotations):
    if 'all' in annotations:
        annotations = _annotations
        annotations.remove('all')
        annotations.remove('none')

    if 'none' in annotations:
        annotations = []
    return annotations


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
        raise ValueError(f"Error opening video file at '{input_path}'")
    
    return video


def read_grayscale_image(input_path: str, filename: str) -> np.ndarray:
    if os.path.isfile(input_path):
        return cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    elif os.path.isdir(input_path) and filename is not None:
        filename = os.path.join(input_path, filename)
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

