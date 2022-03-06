from matplotlib import pyplot as plt
import cv2
from numpy import imag

def convert_to_gray_scale(image):
    """Convert to Gray Scale

    Args:
        image (image): Image to change

    Returns:
        image: Grey Scale Image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def resize_image(image, size= (120,120)):
    """Resizes Image to desired Size

    Args:
        image (image): Image to resize
        size (tuple, optional): size of returned image. Defaults to (120,120).

    Returns:
        image: Resized Image
    """
    resized = cv2.resize(image,size)
    return resized
