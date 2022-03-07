import cv2
import numpy as np

def pyramid(img,scale=1.5, minSize=(30, 30)):
    """ Generates a pyramid of images

    Args:
        img (_type_): image to be processed
        scale (float, optional): Scale factor to which image is divided. Defaults to 1.5.
        minSize (tuple, optional): minimum size of pyramid after which can't be reduced. Defaults to (30, 30).

    Yields:
        _type_: image layer in the pyramid
    """
    yield img
    while True:
        w = int(img.shape[1] / scale)
        img = cv2.resize(img, (w, int(img.shape[0] / scale)))
        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]:
            break
        yield img

def sliding_window(img, stepSize, windowSize):
    """Loop over image with window by stride of stepsize

    Args:
        img (_type_): image to be scanned
        stepSize (_type_): stride of the window
        windowSize (_type_): size of the window

    Yields:
        _type_: array of windows
    """
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield (x, y, img[y:y + windowSize[1], x:x + windowSize[0]])
