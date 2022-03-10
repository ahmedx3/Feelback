import math
import pickle
import cv2
from sklearn.decomposition import PCA

pca = PCA()
class FeatureExtractor:
    def __init__(self, load=False):
        self.pca = None
        if load: self.pca = pickle.load(open('PCAModel.sav', 'rb'))

    def ExtractHOGFeatures(self, img, target_img_size=(32, 32)):
        """Extracts HOG features from an image

        Args:
            img (2D matrix): image to extract features from
            target_img_size (tuple, optional): target image size. Defaults to (32, 32).

        Returns:
            array: HOG features
        """
        
        img = cv2.resize(img, target_img_size)

        cellSize = (3,3)
        blockSize = (6,6)
        nBins = 7

        win_size = (cellSize[1] * cellSize[1], cellSize[0] * cellSize[0])
        block_stride = (cellSize[1], cellSize[0]) 

        hog = cv2.HOGDescriptor(win_size, blockSize, block_stride, cellSize, nBins)
        hog = hog.compute(img)
        hog = hog.flatten()
        return hog

    def TrainPCAonFeatures(self, features, nComponents=None):
        """ Train PCA model on the training features and save model to file

        Args:
            features (2D array): 2D array of features
            nComponents (int, optional): dimensions after reduction. Defaults to None.

        Returns:
            2D array: features after dimensions reduction
        """
        self.pca = PCA(n_components=nComponents)
        self.pca.fit(features)

        filename = 'PCAModel.sav'
        pickle.dump(self.pca, open(filename, 'wb'))

        return self.pca.transform(features)

    def ApplyPCAonFeatures(self, features):
        """ Reduce dimensions of features of sample

        Args:
            features (2D array): features of single or multiple samples 

        Returns:
            2D array: features after dimensions reduction
        """
        return self.pca.transform(features)
