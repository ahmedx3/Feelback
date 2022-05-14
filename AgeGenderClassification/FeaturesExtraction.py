from scipy.signal import convolve2d
import numpy as np
from skimage.feature import local_binary_pattern
import cv2

def extract_LPQ(img, winSize=5, bins=256):
    """Extract LPQ 

    Args:
        img (image): Image to extract from
        winSize (int, optional): Size of Fourier Transform Window. Defaults to 5.

    Returns:
        np.array: Histogram values of LPQ for Image
    """

    # Convert np.image to double
    img = np.float64(img)
    # Calculate radius from window size
    radius = int(winSize / 2 - 0.5)
    # Form spatial coordinates in window
    filters_shape = np.arange(-radius, radius+1)[np.newaxis]

    # Create Filters
    filter_one = np.ones_like(filters_shape)
    filter_two = np.exp(-2*np.pi*filters_shape * (1/winSize)*1j)
    filter_three = np.conj(filter_two)

    # Run filters to compute the frequency response in the four points
    temp_img_filter_two = convolve2d(img, filter_two.T, 'valid')
    temp_img_filter_one = convolve2d(img, filter_one.T, 'valid')
    fourier_response_one = convolve2d(temp_img_filter_one, filter_two, 'valid')
    fourier_response_two = convolve2d(temp_img_filter_two, filter_one, 'valid')
    fourier_response_three = convolve2d(
        temp_img_filter_two, filter_two, 'valid')
    fourier_response_four = convolve2d(
        temp_img_filter_two, filter_three, 'valid')

    # Initilize frequency domain matrix for four frequency coordinates.
    stacked_fourier_response = np.dstack([fourier_response_one.real, fourier_response_one.imag,
                                          fourier_response_two.real, fourier_response_two.imag,
                                          fourier_response_three.real, fourier_response_three.imag,
                                          fourier_response_four.real, fourier_response_four.imag])

    # Perform quantization and compute LPQ codewords
    inds = np.arange(stacked_fourier_response.shape[2])[
        np.newaxis, np.newaxis, :]
    calculated_LPQ = ((stacked_fourier_response > 0)*(2**inds)).sum(2)

    # Normalize histogram
    calculated_LPQ = np.histogram(calculated_LPQ.flatten(), bins=bins)[0]
    calculated_LPQ = calculated_LPQ/calculated_LPQ.sum()

    return calculated_LPQ


def extract_LBP(img, P=8, R=2 , eps=1e-7, bins=256):
    """Extract LBP

    Args:
        img (Image): Image to extract feature
        radius (int, optional): radius around pixel. Defaults to 3.
        eps (_type_, optional): Divide by zero tolerance. Defaults to 1e-7.

    Returns:
        np.array: Numpy Array of histogram values of LBP
    """

    lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=bins)
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def localLBP(image):
    """Taken from the paper Gender Classification Based on Boosting Local Binary Pattern

    Args:
        image (Image): Image to extract feature
    """
    # Check if image is the correct size
    if image.shape != (80, 80):
        image = cv2.resize_image(image, (80, 80))

    feature_vector = []

    # Divide the image into 16 20x20 regions
    for row in [0, 20, 40, 60]:
        for column in [0, 20, 40, 60]:
            cropped_region = image[row:row+20, column:column+20]
            feature = extract_LBP(cropped_region, P=8, R=2 , eps=1e-7, bins=59)
            feature_vector += list(feature)

    feature_vector = np.array(feature_vector)
    return feature_vector

def localLPQ(image):
    """Taken from the paper Gender Classification Based on Boosting Local Binary Pattern

    Args:
        image (Image): Image to extract feature
    """
    # Check if image is the correct size
    if image.shape != (80, 80):
        image = cv2.resize_image(image, (80, 80))

    feature_vector = []

    # Divide the image into 16 20x20 regions
    for row in [0, 20, 40, 60]:
        for column in [0, 20, 40, 60]:
            cropped_region = image[row:row+20, column:column+20]
            feature = extract_LPQ(cropped_region, winSize=3, bins=20)
            feature_vector += list(feature)

    feature_vector = np.array(feature_vector)
    return feature_vector


def extract_features(img, feature="LPQ"):
    """Extracts Features of an image

    Args:
        img (Image): Greyscale image to extract from
        feature (str, optional): Type of feature to extract. Defaults to "LPQ".

    Returns:
        List: feature
    """
    if feature == 'LPQ':
        return extract_LPQ(img, winSize=5)
    elif feature == 'LBP':
        return extract_LBP(img)
    elif feature == 'localLBP':
        return localLBP(img)
    elif feature == 'localLPQ':
        return localLPQ(img)
    return None
