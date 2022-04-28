import cv2
from skimage import exposure

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

def hist_equalize(image):
    """Histogram equalization for image

    Args:
        image (_type_): Image to do the hostogram equalization

    Returns:
        image: Equalized Image
    """
    # equalized = cv2.equalizeHist(image) 
    equalized = exposure.equalize_adapthist(image)
    return equalized


def preprocess_image(img):
    """Preprocess image

    Args:
        img (Image): Image to preprocess

    Returns:
        Image: Preprocessed Image
    """
    
    preprocessed = hist_equalize(img)
    preprocessed = resize_image(preprocessed, (80,80))

    return preprocessed