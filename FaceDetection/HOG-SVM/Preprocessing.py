from matplotlib import pyplot as plt
import cv2

def HistogramEqualization(grayScaleImage):
    """equalize histogram of grayscale image

    Args:
        grayScaleImage (_type_): grayscale image to be equalized

    Returns:
        _type_: equalized grayscale image
    """
    equlizedImage = cv2.equalizeHist(grayScaleImage)
    return equlizedImage

def ConvertToGrayScale(image):
    """convert image to grayscale

    Args:
        image (_type_): image to be converted

    Returns:
        _type_: gray scale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

