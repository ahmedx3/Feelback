from scipy.signal import convolve2d
import numpy as np
from skimage.feature import local_binary_pattern

def extract_LPQ(img, winSize=5):
    """Extract LPQ 

    Args:
        img (image): Image to extract from
        winSize (int, optional): Size of Fourier Transform Window. Defaults to 5.

    Returns:
        np.array: Histogram values of LPQ for Image
    """

    img = np.float64(img)  # Convert np.image to double
    radius = int(winSize / 2 - 0.5)  # Get radius from window size
    # Form spatial coordinates in window
    x = np.arange(-radius, radius+1)[np.newaxis]

    w0 = np.ones_like(x)
    w1 = np.exp(-2*np.pi*x * (1/winSize)*1j)
    w2 = np.conj(w1)

    # Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    fr_1 = convolve2d(convolve2d(img, w0.T, 'valid'), w1, 'valid')
    fr_2 = convolve2d(convolve2d(img, w1.T, 'valid'), w0, 'valid')
    fr_3 = convolve2d(convolve2d(img, w1.T, 'valid'), w1, 'valid')
    fr_4 = convolve2d(convolve2d(img, w1.T, 'valid'), w2, 'valid')

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([fr_1.real, fr_1.imag,
                          fr_2.real, fr_2.imag,
                          fr_3.real, fr_3.imag,
                          fr_4.real, fr_4.imag])

    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0)*(2**inds)).sum(2)

    # Normalize histogram
    LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]
    LPQdesc = LPQdesc/LPQdesc.sum()

    return LPQdesc

def extract_LBP(img, radius=3, eps=1e-7):
    """Extract LBP

    Args:
        img (Image): Image to extract feature
        radius (int, optional): radius around pixel. Defaults to 3.
        eps (_type_, optional): Divide by zero tolerance. Defaults to 1e-7.

    Returns:
        np.array: Numpy Array of histogram values of LBP
    """

    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, 'default')
    (hist, _) = np.histogram(lbp.ravel(), bins=256)
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


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
        return extract_LBP(img, radius=3, eps=1e-7)
    return None