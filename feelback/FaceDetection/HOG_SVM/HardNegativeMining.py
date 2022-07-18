# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import pickle
from .Utils import *
from .FeaturesExtraction import *
import os
from .SlidingWindow import pyramid

def slidingWindow(img, stepSize, windowSize):
    windowsArr = []
    for y in range(0, img.shape[0] - windowSize[0] + 1, stepSize):
        for x in range(0, img.shape[1] - windowSize[1] + 1, stepSize):
            windowsArr.append(((x, y), img[y:y + windowSize[1], x:x + windowSize[0]]))
    return windowsArr

# Load Model
modelName = "./Models/Model_v3.sav"
model = pickle.load(open(modelName, 'rb'))

pca = pickle.load(open("./Models/PCA_v3.sav", 'rb'))

################################## Hyperparameters ##################################
(winW, winH) = (19, 19)  # window width and height
pyramidScale = 10  # Scale factor for the pyramid
stepSize = 2  # Step size for the sliding window

# extract image negative examples from the dataset folder
def load_dataset_negative_images(path_to_dataset):
    uniqueIdentifier = 0
    negativeImages = []
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)
    directoriesNames = os.listdir(path_to_dataset)
    print(directoriesNames)
    for directory in directoriesNames:
        print(directory)
        directoryPath = os.path.join(path_to_dataset, directory)
        for root, dirs, files in os.walk(directoryPath):
            for file in files:
                # append the file name to the list
                fileName = os.path.join(root, file)
                # read the image
                img = cv2.imread(fileName)
                
                windows = slidingWindow(img, stepSize, (winW, winH))
                if (len(windows)) == 0:
                    break
                # print("[INFO] Num of windows in the current image pyramid ",len(windows))

                indices, patches = zip(*windows)
                patches_hog = []
                for patch in patches:
                    image_features = ExtractHOGFeatures(patch,flatten=True)
                    image_features = ApplyPCA(image_features,pca)
                    predicted_label = model.predict([image_features])
                    if predicted_label == "Faces":
                        cv2.imwrite("./Data/Non/Mined2/" + str(uniqueIdentifier) + ".jpg", patch)
                        uniqueIdentifier += 1
                        print("[INFO] Image saved " + str(uniqueIdentifier))    


load_dataset_negative_images("./HardNegative/File")
