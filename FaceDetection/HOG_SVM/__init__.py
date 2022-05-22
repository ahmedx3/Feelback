# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import pickle as pickle
from .FeaturesExtraction import *
from .SlidingWindow import *
from .Utils import *
import cv2


class FaceDetector:
    def __init__(self, model_path, pca_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.pca = pickle.load(open(pca_path, 'rb'))

    def detect(self, frame):
        ################################## Hyperparameters ##################################
        (winW, winH) = (19, 19)  # window width and height
        pyramidScale = 1.5  # Scale factor for the pyramid
        stepSize = 2  # Step size for the sliding window
        overlappingThreshold = 0.3  # Overlap threshold for non-maximum suppression
        skinThreshold = 0.4  # threshold for skin color in the window
        edgeThreshold = 0.2  # threshold for edge percentage in the window
        resizeFactor = 8
        #####################################################################################

        frameOriginal = frame.copy()
        frame = cv2.resize(frame, (int(frame.shape[1] / resizeFactor), int(frame.shape[0] / resizeFactor)))
        faces = []

        for image in pyramid(frame, pyramidScale, minSize=(30, 30)):

            scaleFactor = frameOriginal.shape[0] / float(image.shape[0])
            skinMask = DetectSkinColor(image)
            edges = EdgeDetection(image)
            commonMask = detectCommonMask(edges, skinMask)
            windows = slidingWindow(image, stepSize, (winW, winH), skinMask, edges, commonMask, skinThreshold,
                                    edgeThreshold)
            if (len(windows)) == 0:
                break
            indices, patches = zip(*windows)
            grayScaledPatches = [HistogramEqualization(patch) for patch in patches]
            hogFeatures = vectorizedHogSlidingWindows(grayScaledPatches)
            patches_hog = []
            for i in range(len(windows)):
                patches_hog.append(ApplyPCA(hogFeatures[:, :, :, :, :, i].flatten(), self.pca))
            predicted_label = self.model.predict(patches_hog)
            indices = np.array(indices)
            for i, j in indices[predicted_label == "Faces"]:
                faces.append((int(i * scaleFactor), int(j * scaleFactor), int((i + winW) * scaleFactor),
                              int((j + winH) * scaleFactor)))

        faces = nonMaxSuppression(faces, overlappingThreshold)

        # Increase the size of the bounding boxes
        # biggerFaces = []
        # resizeRatio = 5 # the smaller the number, the bigger the bounding box
        # for (x1, y1, x2, y2) in faces:
        #     squareLen = x2 - x1
        #     biggerFaces.append((x1 - squareLen/resizeRatio, y1 - squareLen/resizeRatio, x2 + squareLen/resizeRatio, y2 + squareLen/resizeRatio))

        return faces

# FaceDetector = FaceDetector()
# faces = FaceDetector.detect(cv2.imread('D:/Git/Feelback/FaceDetection/HOG_SVM/Examples/Test8.jpg'))
# print(faces)
