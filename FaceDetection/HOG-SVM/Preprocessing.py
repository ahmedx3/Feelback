from matplotlib import pyplot as plt
import cv2

def HistogramEqualization(grayScaleImage):
    """equalize histogram of grayscale image

    Args:
        grayScaleImage (_type_): grayscale image to be equalized

    Returns:
        _type_: equalized grayscale image
    """
    gray = cv2.cvtColor(grayScaleImage, cv2.COLOR_BGR2GRAY)
    equlizedImage = cv2.equalizeHist(gray)
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

def DetectSkinColor(img):
    """Detect skin color in an image

    Args:
        img (_type_): image

    Returns:
        _type_: a boolean mask of the image skin
    """
    # extract red and green and blue channels
    b, g, r = cv2.split(img)

    # do a piecewise log
    calc1 = cv2.log(r/g)
    calc2 = cv2.log(b/g)

    # mask if test1 between [0.15;1.1] and test2 between [-4;0.3]
    mask = (calc1 > 0.15) & (calc1 < 1.1) & (calc2 > -4) & (calc2 < 0.3)
    
    return mask