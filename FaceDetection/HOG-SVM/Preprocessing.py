from matplotlib import pyplot as plt
import cv2

def HistogramEqualization(grayScaleImage):
    """
    Histogram Equalization
    """
    equlizedImage = cv2.equalizeHist(grayScaleImage)
    return equlizedImage

def ConvertToGrayScale(image):
    """
    Convert to Gray Scale
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

