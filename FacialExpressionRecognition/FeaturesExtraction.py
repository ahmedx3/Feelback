import math
import pickle
import cv2
from matplotlib.patches import Rectangle
import dlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import FacialExpressionRecognition.Utils as Utils
import pywt

class FeatureExtractor:
    def __init__(self, load=False):
        self.flag = True
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('FacialExpressionRecognition/Experiments/shape_predictor_68_face_landmarks.dat')
        if load:
            self.pca = pickle.load(open('FacialExpressionRecognition/Models/PCAModel.sav', 'rb'))
            self.scaler = pickle.load(open('FacialExpressionRecognition/Models/scalerModel.sav', 'rb'))

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

        filename = 'Models/PCAModel.sav'
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

    def ExtractLandMarks_method1(self,img, target_img_size=(128,128)):
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

    def ExtractLandMarks_method2(self,img, target_img_size=(128,128)):
        """ Extract landmarks from face then calculate geometric features from them

        Args:
            img (2D-arry): Image of faces
            target_img_size (tuple, optional): size of image after resizing. Defaults to (128,128).

        Returns:
            array: Extracted features array of length 72
        """
        # img = cv2.resize(img, target_img_size)

        # Detect the face
        rects = self.detector(img, 1)

        # Detect landmarks for each face
        for rect in rects:

            # Crop and resize the faces
            cropped = img[max(rect.top(), 0):min(rect.bottom()+1, img.shape[0]), max(rect.left(), 0):min(rect.right()+1, img.shape[1])]
            return self.ExtractLandMarks(cropped)
        return None

    def ExtractLandMarks(self, cropped, target_img_size=(128,128)):
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
        
        features = []
        for i in [17,18,19,20,21,37]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(36).x, shape.part(i).y, shape.part(36).y))
        
        for i in [18,19,20,21]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(39).x, shape.part(i).y, shape.part(39).y))
        
        for i in [19,20,21,22,23,24,39,42]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(27).x, shape.part(i).y, shape.part(27).y))
        
        for i in [22,23,24,25]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(42).x, shape.part(i).y, shape.part(42).y))
        
        for i in [22,23,24,25,26,44]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(45).x, shape.part(i).y, shape.part(45).y))
        
        features.append(self.eculidianDistance(shape.part(37).x, shape.part(41).x, shape.part(37).y, shape.part(41).y))
        
        for i in [37,38,39]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(40).x, shape.part(i).y, shape.part(40).y))
        
        for i in [42,43,44]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(47).x, shape.part(i).y, shape.part(47).y))
        
        features.append(self.eculidianDistance(shape.part(44).x, shape.part(46).x, shape.part(44).y, shape.part(46).y))
        features.append(self.eculidianDistance(shape.part(32).x, shape.part(40).x, shape.part(32).y, shape.part(40).y))
        features.append(self.eculidianDistance(shape.part(34).x, shape.part(47).x, shape.part(34).y, shape.part(47).y))
        
        for i in [31,33]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(50).x, shape.part(i).y, shape.part(50).y))
        for i in [33,35]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(52).x, shape.part(i).y, shape.part(52).y))
        
        features.append(self.eculidianDistance(shape.part(33).x, shape.part(51).x, shape.part(33).y, shape.part(51).y))
        
        for i in [49,51]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(61).x, shape.part(i).y, shape.part(61).y))
        for i in [51,53]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(63).x, shape.part(i).y, shape.part(63).y))
        
        for i in [17,36,40,31,49]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(48).x, shape.part(i).y, shape.part(48).y))
        
        features.append(self.eculidianDistance(shape.part(60).x, shape.part(62).x, shape.part(60).y, shape.part(62).y))
        features.append(self.eculidianDistance(shape.part(64).x, shape.part(62).x, shape.part(64).y, shape.part(62).y))
        
        for i in [53,35,47,45,26]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(54).x, shape.part(i).y, shape.part(54).y))
        
        for i in [48,67]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(59).x, shape.part(i).y, shape.part(59).y))
        
        for i in [60,50,62,52,64]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(66).x, shape.part(i).y, shape.part(66).y))
        
        for i in [65,54]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(55).x, shape.part(i).y, shape.part(55).y))
        
        features.append(self.eculidianDistance(shape.part(58).x, shape.part(62).x, shape.part(58).y, shape.part(62).y))
        
        for i in [67,65]:
            features.append(self.eculidianDistance(shape.part(i).x, shape.part(57).x, shape.part(i).y, shape.part(57).y))
        
        features.append(self.eculidianDistance(shape.part(56).x, shape.part(62).x, shape.part(56).y, shape.part(62).y))
        
        return np.array(features)

    def trainNormalizeFeatures(self, features):
        """ Standardize (Normalize) the feature vector then save the model to file

        Args:
            features (2D-array): Features array of the training samples

        Returns:
            2D-array: Features after standardization
        """
        scaler = StandardScaler()
        scaler.fit(features)

        filename = 'Models/scalerModel.sav'
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

    def DWT(self, img, target_img_size=(128,128)):
         # Detect the face
        rects = self.detector(img, 1)

        # Detect landmarks for each face
        for rect in rects:
            # Crop and resize the faces
            cropped = img[max(rect.top(), 0):min(rect.bottom()+1, img.shape[0]), max(rect.left(), 0):min(rect.right()+1, img.shape[1])]
            cropped = cv2.resize(cropped, target_img_size)
            (_, cD) = pywt.dwt(cropped, 'db1')
            return np.array(cD).flatten()
        return None

    def GaborFilter(self, img, target_img_size=(128,128)):
         # Detect the face
        rects = self.detector(img, 1)

        # Detect landmarks for each face
        for rect in rects:
            # Crop and resize the faces
            cropped = img[max(rect.top(), 0):min(rect.bottom()+1, img.shape[0]), max(rect.left(), 0):min(rect.right()+1, img.shape[1])]
            cropped = cv2.resize(cropped, target_img_size)
            gabor1 = cv2.getGaborKernel((18,18), 1.5,math.pi/4,5,1.5, 0)
            gabor2 = cv2.getGaborKernel((18,18), 1.5,3*math.pi/4,5,1.5, 0)
            filtered = cv2.filter2D(cropped,-1,gabor1)
            filtered = cv2.filter2D(filtered,-1,gabor2)
            return np.array(filtered).flatten()
        return None
    
    def extract_features(self, img, feature='LANDMARKS'):
        if feature == 'LANDMARKS':
            return self.normalizeFeatures(self.ExtractLandMarks(img).reshape(1,-1))[0]
        if feature == 'HOG':
            feat = self.ExtractHOGFeatures(img)
            return self.ApplyPCAonFeatures(feat.reshape(1,-1))[0]
        if feature == 'GABOR':
            feat = self.GaborFilter(img)
            return self.ApplyPCAonFeatures(feat)


