from builtins import print as _print
import cv2
import numpy as np


__VERBOSE__ = False


def imshow(image: np.ndarray, title=None, delay=0):
    if __VERBOSE__:
        if title is not None:
            cv2.imshow(f"DEBUG: {title}", image)
        else:
            cv2.imshow("DEBUG", image)
        cv2.waitKey(delay)


def print(*args, sep=' ', end='\n'):
    if __VERBOSE__:
        _print(*args, sep=sep, end=end, flush=True)

