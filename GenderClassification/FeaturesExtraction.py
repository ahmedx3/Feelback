from scipy.signal import convolve2d
import numpy as np

def extractLPQ(img, winSize=5):

    img = np.float64(img)  # Convert np.image to double
    radius = (winSize-1)/2  # Get radius from window size
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