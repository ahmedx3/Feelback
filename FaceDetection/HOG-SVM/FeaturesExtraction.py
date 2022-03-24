import cv2
from Utils import *

def ExtractHOGFeatures(img,target_img_size=(19,19)):
    """
    Extracts HOG features from an image
    :param img: image to extract features from
    :param cellSize: size of a cell
    :param blockSize: size of a block
    :param nBins: number of bins
    :return: HOG features
    """
    img = cv2.resize(img, target_img_size)
    img = HistogramEqualization(img)
    
    # hog = getHOG(img)

    cellSize = (3,3)
    blockSize = (6,6)
    nBins = 7

    win_size = (cellSize[1] * cellSize[1], cellSize[0] * cellSize[0])
    block_stride = (cellSize[1], cellSize[0]) 

    hog = cv2.HOGDescriptor(win_size, blockSize, block_stride, cellSize, nBins)
    hog = hog.compute(img)
    hog = hog.flatten()

    return hog

def ApplyPCA(features,pca):
    """
    Applies PCA to a feature vector
    :param feature: feature vector
    :param pca: PCA object
    :return: PCA transformed feature vector
    """
    features = np.array(features)
    return pca.transform(features.reshape(1,-1))[0]