import math
import pickle
import cv2
from matplotlib.patches import Rectangle
import dlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import Utils

class FeatureExtractor:
    def __init__(self, load=False):
        self.flag = True
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('Experiments/shape_predictor_68_face_landmarks.dat')
        if load:
            self.pca = pickle.load(open('PCAModel.sav', 'rb'))
            self.scaler = pickle.load(open('scalerModel.sav', 'rb'))

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
        pca = PCA(n_components=nComponents)
        pca.fit(features)

        filename = 'PCAModel.sav'
        pickle.dump(pca, open(filename, 'wb'))

        return pca.transform(features)

    def ApplyPCAonFeatures(self, features):
        """ Reduce dimensions of features of sample

        Args:
            features (2D array): features of single or multiple samples 

        Returns:
            2D array: features after dimensions reduction
        """
        return self.pca.transform(features)

    def eculidianDistance(self,x1,x2,y1,y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def ExtractLandMarks(self,img, target_img_size=(128,128)):
        """ Extract landmarks from face then calculate geometric features from them

        Args:
            img (2D-arry): Image of faces
            target_img_size (tuple, optional): size of image after resizing. Defaults to (128,128).

        Returns:
            array: Extracted features array of length 22 (8 points (x, y) and 6 distances)
        """
        # img = cv2.resize(img, target_img_size)

        # Detect the face
        rects = self.detector(img, 1)

        # Detect landmarks for each face
        for rect in rects:

            # Crop and resize the faces
            cropped = img[max(rect.top(), 0):min(rect.bottom()+1, img.shape[0]), max(rect.left(), 0):min(rect.right()+1, img.shape[1])]
            cropped = cv2.resize(cropped, target_img_size)
            box = dlib.rectangle(0,0,cropped.shape[1],cropped.shape[0])
            
            # Get the landmark points    
            shape = self.predictor(cropped, box)

            # Just for Testing TODO:Remove later 
            # if self.flag:
            #     print(rect.top(),rect.bottom(), rect.left(),rect.right())
            #     # Display the image
            #     image = img[:,:]
            #     cv2.rectangle(image, (rect.left(), rect.top()),(rect.right(), rect.bottom()), (0, 0, 255), 1)
            #     Utils.show_image(cropped, 'Landmark Detection')
            #     self.flag = False
            
            keypoints = [19,24,39,42,48,51,54,57]
            
            features = np.zeros(22, dtype="float")
            
            features[0] = (self.eculidianDistance(shape.part(19).x, shape.part(39).x, shape.part(19).y, shape.part(39).y)
                        + self.eculidianDistance(shape.part(24).x, shape.part(42).x, shape.part(24).y, shape.part(42).y))/(2)

            features[1] = (self.eculidianDistance(shape.part(39).x, shape.part(51).x, shape.part(39).y, shape.part(51).y)
                        + self.eculidianDistance(shape.part(42).x, shape.part(51).x, shape.part(42).y, shape.part(51).y))/(2)

            features[2] = self.eculidianDistance(shape.part(48).x, shape.part(54).x, shape.part(48).y, shape.part(54).y)
            features[3] = self.eculidianDistance(shape.part(51).x, shape.part(57).x, shape.part(51).y, shape.part(57).y)

            avrgPointx = (shape.part(48).x + shape.part(54).x) / 2
            avrgPointy = (shape.part(48).y + shape.part(54).y) / 2
            features[4] = self.eculidianDistance(shape.part(51).x, avrgPointx, shape.part(51).y, avrgPointy)
            features[5] = self.eculidianDistance(shape.part(57).x, avrgPointx, shape.part(57).y, avrgPointy)

            j = 6
            for i in keypoints:
                features[j] = shape.part(i).x 
                features[j+1] = shape.part(i).y
                j+=2
            
            return features
        return None

    def trainNormalizeFeatures(self, features):
        """ Standardize (Normalize) the feature vector then save the model to file

        Args:
            features (2D-array): Features array of the training samples

        Returns:
            2D-array: Features after standardization
        """
        scaler = StandardScaler()
        scaler.fit(features)

        filename = 'scalerModel.sav'
        pickle.dump(scaler, open(filename, 'wb'))
        return scaler.transform(features)
    
    def normalizeFeatures(self, features):
        """ Standardize (Normalize) the feature vector with the already trained model

        Args:
            features (2D-array): Features array of the test samples

        Returns:
            2D-array: Features after standardization
        """
        return self.scaler.transform(features)